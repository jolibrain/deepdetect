/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Part of this code is adapted from Caffe's convert_imageset tool
 */

#include "caffeinputconns.h"
#include "utils/utils.hpp"
#include <boost/multi_array.hpp>
#include <algorithm>
#include <random>
#ifdef USE_HDF5
#include <H5Cpp.h>
#endif // USE_HDF5
#include <memory>
#include "utf8.h"

using namespace caffe;

namespace dd
{
  
  void CaffeInputInterface::write_class_weights(const std::string &model_repo,
						const APIData &ad_mllib)
  {
    std::string cl_file = model_repo + "/class_weights.binaryproto";
    if (ad_mllib.has("class_weights"))
      {
	std::vector<double> cw;
	try
	  {
	    cw = ad_mllib.get("class_weights").get<std::vector<double>>();
	  }
	catch (std::exception &e)
	  {
	    // let's try array of int, that's a common mistake
	    std::vector<int> cwi = ad_mllib.get("class_weights").get<std::vector<int>>();
	    for (int v: cwi)
	      cw.push_back(static_cast<double>(v));
	  }
	int nclasses = cw.size();
	BlobProto cw_blob;
	cw_blob.set_num(1);
	cw_blob.set_channels(1);
	cw_blob.set_height(nclasses);
	cw_blob.set_width(nclasses);
	for (int i=0;i<nclasses;i++)
	  {
	    for (int j=0;j<nclasses;j++)
	      {
		if (i == j)
		  cw_blob.add_data(cw.at(i));
		else cw_blob.add_data(0.);
	      }
	  }
	//_logger->info("write class weights to {}",cl_file);
	WriteProtoToBinaryFile(cw_blob,cl_file.c_str());
      }
  }

  std::string ImgCaffeInputFileConn::guess_encoding(const std::string &file)
  {
    size_t p = file.rfind('.');
    if (p == file.npos)
      _logger->warn("Failed to guess the encoding of {}",file);
    std::string enc = file.substr(p);
    std::transform(enc.begin(),enc.end(),enc.begin(),::tolower);
    return enc;
  }
  
  // convert images into db entries
  // a path must:
  // - either contain directories as classes holding image
  //   files for each class. The name of the class is the name of the directory.
  // - either be a file containing in each line the path to an image
  //   and the label associated
  int ImgCaffeInputFileConn::images_to_db(const std::vector<std::string> &rpaths,
					  const std::string &traindbname,
					  const std::string &testdbname,
					  const bool &folders,
					  const std::string &backend,
					  const bool &encoded,
					  const std::string &encode_type)
  {
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;
    
    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    if (fileops::file_exists(dbfullname))
      {
	_logger->warn("image db file {} already exists, bypassing creation but checking on records",dbfullname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	_db_batchsize = db->Count();
	_logger->warn("image db file {} with {} records",dbfullname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    _logger->warn("image db file {} already exists, bypassing creation but checking on records",testdbfullname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->warn("image db file {} with {} records",testdbfullname,_db_testbatchsize);
	  }
	return 0;
      }

    // list files and classes, possibly shuffle / split them
    int cl = 0;
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    std::unordered_map<std::string,int> hcorresp_r; // reverse correspondence for test set.
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
		
    // If the input is a folder
    if (folders)
      {
      // list directories in dataset train folder
      std::unordered_set<std::string> subdirs;
      if (fileops::list_directory(rpaths.at(0),false,true,false,subdirs))
        throw InputConnectorBadParamException("failed reading image train data directory " + rpaths.at(0));

      auto uit = subdirs.begin();
      while(uit!=subdirs.end())
        {
        std::unordered_set<std::string> subdir_files;
        if (fileops::list_directory((*uit),true,false,true,subdir_files))
          throw InputConnectorBadParamException("failed reading image train data sub-directory " + (*uit));
        std::string cls = dd_utils::split((*uit),'/').back();
        hcorresp.insert(std::pair<int,std::string>(cl,cls));
        hcorresp_r.insert(std::pair<std::string,int>(cls,cl));
        auto fit = subdir_files.begin();
        while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
          {
          lfiles.push_back(std::pair<std::string,int>((*fit),cl));
          ++fit;
          }
        ++cl;
        ++uit;
        }
      }
    // Else if input is a file
    else
      {
      std::ifstream infile(rpaths.at(0));
      std::string line;
      int line_num = 1;
      int cl = 0;
      while (std::getline(infile, line))
        {
        std::istringstream iss(line);
        string filename;
        string label;
        iss >> filename >> label;

        int label_int = cl;
        std::unordered_map<std::string,int>::const_iterator it;
        // Check if mapping does not exist
        if ((it=hcorresp_r.find(label))==hcorresp_r.end())
          {
          hcorresp.insert(std::pair<int,std::string>(cl,label));
          hcorresp_r.insert(std::pair<std::string,int>(label,cl));
          cl++;
          }
        else
          label_int = it->second;

        line_num ++;
        // Append lfiles
        lfiles.push_back(std::pair<std::string,int>(filename, label_int));

        }
      }
	
    if (_shuffle)
      {
	std::mt19937 g;
	if (_seed >= 0)
	  g = std::mt19937(_seed);
	else
	  {
	    std::random_device rd;
	    g = std::mt19937(rd());
	  }
	std::shuffle(lfiles.begin(),lfiles.end(),g);
      }
    
    // test split
    std::vector<std::pair<std::string,int>> test_lfiles;
    if (_test_split > 0.0)
      {
	int split_size = std::floor(lfiles.size() * (1.0-_test_split));
	auto chit = lfiles.begin();
	auto dchit = chit;
	int cpos = 0;
	while(chit!=lfiles.end())
	  {
	    if (cpos == split_size)
	      {
		if (dchit == lfiles.begin())
		  dchit = chit;
		test_lfiles.push_back((*chit));
	      }
	    else ++cpos;
	    ++chit;
	  }
	lfiles.erase(dchit,lfiles.end());
      }
    else if (rpaths.size() > 1)
      {
      // If input is a folder
      if (folders)
        {
        // list directories in dataset test folder
        std::unordered_set<std::string> test_subdirs;
        if (fileops::list_directory(rpaths.at(1),false,true,false,test_subdirs))
          throw InputConnectorBadParamException("failed reading image test data directory " + rpaths.at(1));

        // list files and classes, possibly shuffle / split them
        std::unordered_map<std::string,int>::const_iterator hcit;
        auto uit = test_subdirs.begin();
        while(uit!=test_subdirs.end())
          {
          std::unordered_set<std::string> subdir_files;
          if (fileops::list_directory((*uit),true,false,true,subdir_files))
            throw InputConnectorBadParamException("failed reading image test data sub-directory " + (*uit));
          std::string cls = dd_utils::split((*uit),'/').back();
          if (!_autoencoder && (hcit=hcorresp_r.find(cls))==hcorresp_r.end())
            {
	      _logger->error("class {} appears in testing set but not in training set, skipping",cls);
	      ++uit;
	      continue;
            }
          int cl = _autoencoder ? 0 : (*hcit).second;
          auto fit = subdir_files.begin();
          while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
	          {
            test_lfiles.push_back(std::pair<std::string,int>((*fit),cl));
            ++fit;
            }
          ++uit;
          }
        }
      // Else if input is a file
      else
        {
        std::ifstream infile(rpaths.at(0));
        std::string line;
        int line_num = 1;
        while (std::getline(infile, line))
          {
          std::istringstream iss(line);
          string filename;
          string label;
          iss >> filename >> label;

          // Check if mapping does not exist, go to next file
          std::unordered_map<std::string,int>::const_iterator it;
          if (!_autoencoder && (it=hcorresp_r.find(label))==hcorresp_r.end())
            {
	      _logger->error("class {} appears in testing set but not in training set, skipping",label);
	      continue;
            }
          line_num++;
          // Append lfiles
          test_lfiles.push_back(std::pair<std::string,int>(filename, it->second));
          }
        }
      }
    _db_batchsize = lfiles.size();
    _db_testbatchsize = test_lfiles.size();
    
    _logger->info("a total of {} images",lfiles.size());
    if (lfiles.empty())
      throw InputConnectorBadParamException("no image data found in repository");
    
    // write files to dbs (i.e. train and possibly test)
    write_image_to_db(dbfullname,lfiles,backend,encoded,encode_type);
    if (!test_lfiles.empty())
      write_image_to_db(testdbfullname,test_lfiles,backend,encoded,encode_type);

    // write corresp file
    std::ofstream correspf(_model_repo + "/" + _correspname,std::ios::binary);
    auto hit = hcorresp.begin();
    while(hit!=hcorresp.end())
      {
	correspf << (*hit).first << " " << (*hit).second << std::endl;
	++hit;
      }
    correspf.close();
    
    return 0;
  }


  void ImgCaffeInputFileConn::create_test_db_for_imagedatalayer(const std::string &test_lst,
								const std::string &testdbname,
								const std::string &backend,
								const bool &encoded,
								const std::string &encode_type) // 'png', 'jpg', ...
{
  std::string testdbfullname = testdbname + "." + backend;
  if (fileops::file_exists(testdbfullname))
    {
      _logger->warn("test db {} already exists, bypassing creation",testdbfullname);
      return;
    }
  vector<std::pair<std::string, std::vector<float> > > lines;
  try
    {
      caffe::ReadImagesList(test_lst,&lines);
    }
  catch (std::exception &e)
    {
      _logger->error("failed reading image list for image data layer");
      throw InputConnectorBadParamException("failed reading image list for image data layer");
    }
  if (lines.empty())
    {
      _logger->error("empty data from {}",test_lst);
      throw InputConnectorBadParamException("empty data from " + test_lst);
    }
    std::vector<std::pair<std::string,int>> test_lfiles_1;
    std::vector<std::pair<std::string,std::vector<float>>> test_lfiles_n;
    int nlabels = (*lines.begin()).second.size();
    for (auto line: lines)
      {
	if (nlabels == 1) // XXX: expected to be 1 for all samples, variable label size not yet allowed
	  test_lfiles_1.push_back(std::pair<std::string,float>(_root_folder+line.first,line.second.at(0)));
	else test_lfiles_n.push_back(std::pair<std::string,std::vector<float>>(_root_folder+line.first,line.second));
      }
    if (nlabels == 1)
      write_image_to_db(testdbfullname,test_lfiles_1,backend,encoded,encode_type);
    else write_image_to_db_multilabel(testdbfullname,test_lfiles_n,backend,encoded,encode_type);
  }


  void ImgCaffeInputFileConn::write_image_to_db(const std::string &dbfullname,
						const std::vector<std::pair<std::string,int>> &lfiles,
						const std::string &backend,
						const bool &encoded,
						const std::string &encode_type)
  {
    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());
    
    // Storing to db
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    for (int line_id = 0; line_id < (int)lfiles.size(); ++line_id) {
      Datum datum;
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
	enc = guess_encoding(lfiles[line_id].first);
      }
      else if (!encoded)
	enc = "";

      status = ReadImageToDatum(lfiles[line_id].first,
				lfiles[line_id].second, _height, _width, !_bw,
				enc, &datum, this->_unchanged_data);
      if (status == false) continue;
      
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
			    lfiles[line_id].first.c_str());

      // put in db
      std::string out;
      if(!datum.SerializeToString(&out))
	_logger->error("Failed serialization of datum for db storage");
      txn->Put(string(key_cstr, length), out);
      
      if (++count % 1000 == 0) {
	// commit db
	txn->Commit();
	txn.reset(db->NewTransaction());
	_logger->info("Processed {} files",count);
      }
    }
    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      _logger->info("Processed {} files",count);
    }
  }

  void ImgCaffeInputFileConn::write_image_to_db_multilabel(const std::string &dbfullname,
							   const std::vector<std::pair<std::string,std::vector<float>>> &lfiles,
							   const std::string &backend,
							   const bool &encoded,
							   const std::string &encode_type)
  {
    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());
    
    // Storing to db
    //Datum datum;
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    for (int line_id = 0; line_id < (int)lfiles.size(); ++line_id) {
      Datum datum;
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
	enc = guess_encoding(lfiles[line_id].first);
      }
      status = ReadImageToDatum(lfiles[line_id].first,
				lfiles[line_id].second[0], _height, _width, !_bw, // XXX: passing first label, fixing labels below
				enc, &datum);
      if (status == false)
	_logger->error("failed reading image {}",lfiles[line_id].first);
      
      // store multi labels into float_data in the datum (encoded image should be into data as bytes)
      std::vector<float> labels = lfiles[line_id].second;
      for (auto l: labels)
	{
	  datum.add_float_data(l);
	}
      
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
			    lfiles[line_id].first.c_str());
      
      // put in db
      std::string out;
      if(!datum.SerializeToString(&out))
	_logger->error("Failed serialization of datum for db storage");
      txn->Put(string(key_cstr, length), out);
      
      if (++count % 1000 == 0) {
	// commit db
	txn->Commit();
	txn.reset(db->NewTransaction());
	_logger->info("Processed {} files",count);
      }
    }
    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      _logger->info("Processed {} files",count);
    }
  }

  // - fixed size in-memory arrays put down to disk at once
#ifdef USE_HDF5
  void ImgCaffeInputFileConn::images_to_hdf5(const std::vector<std::string> &img_lists,
					     const std::string &dbfullname,
					     const std::string &test_dbfullname)
  {
    
    // test whether dbs already exist
    if (fileops::file_exists(dbfullname + "_0.h5"))
      {	
	if (!fileops::file_exists(_model_repo + "/" + _correspname))
	  throw InputConnectorBadParamException("found h5 db but no corresp.txt file, erase h5 to rebuild them instead ?");
	std::ifstream in(_model_repo + "/" + _correspname);
	if (!in.is_open())
	  throw InputConnectorBadParamException("failed opening corresp.txt file");
	int nlines = 0;
	std::string line;
	while(getline(in,line))
	  ++nlines;
	_alphabet_size = nlines;

	if (!fileops::file_exists(_model_repo + "/testing.txt"))
	  _logger->info("no hdf5 test db list found, no test set");
	else
	  {
	    std::string tfilename;
	    int tsize = 0;
        std::ifstream in(_model_repo + "/testing.txt");
	    while(getline(in,tfilename))
	      {
		H5::H5File tfile(tfilename, H5F_ACC_RDONLY);
		H5::DataSet dataset = tfile.openDataSet("label");
		//H5::FloatType datatype = dataset.getFloatType();
		//tsize += datatype.getSize();
		
		H5::DataSpace dataspace = dataset.getSpace();
		hsize_t dims[2];
		dataspace.getSimpleExtentDims(dims,NULL);
		tsize += dims[0];
	      }
	    _db_testbatchsize = tsize;
	    _logger->info("hdf5 test set size={}",tsize);
	  }
	return;
      }
	
    //TODO: read / shuffle / split list of images
    
    std::unordered_map<uint32_t,int> alphabet;
    alphabet[0] = 0; // space character
    int max_ocr_length = -1;

    std::string train_list = _model_repo + "/training.txt";
    write_images_to_hdf5(img_lists.at(0), dbfullname, train_list, alphabet, max_ocr_length, true);
    _logger->info("ctc alphabet training size={}",alphabet.size());
    
    if (img_lists.size() > 1)
      {
	std::string test_list = _model_repo + "/testing.txt";
	write_images_to_hdf5(img_lists.at(1), test_dbfullname, test_list, alphabet, max_ocr_length, false);
      }
    
    // save the alphabet as corresp file
    std::ofstream correspf(_model_repo + "/" + _correspname,std::ios::binary);
    auto hit = alphabet.begin();
    while(hit!=alphabet.end())
      {
	correspf << (*hit).second << " " << std::to_string((*hit).first) << std::endl;
	++hit;
      }
    correspf.close();
    _alphabet_size = alphabet.size();
  }

  void ImgCaffeInputFileConn::write_images_to_hdf5(const std::string &inputfilename,
						   const std::string &dbfullname,
						   const std::string &dblistfilename,
						   std::unordered_map<uint32_t,int> &alphabet,
						   int &max_ocr_length,
						   const bool &train_db)
  {
    std::ifstream train_file(inputfilename);
    std::string line;
    std::unordered_map<uint32_t,int>::iterator ait;
    
    // count file lines, we're using fixed-size in-memory array due to
    // complexity of hdf5 handling of incremental datasets
    int clines = 0;
    while(std::getline(train_file, line))
      {
	std::vector<std::string> elts = dd_utils::split(line,' ');
	if (train_db)
	  {
	    int ocr_size = 0;
	    for (size_t k=1;k<elts.size();k++)
	      {
		ocr_size += elts.at(k).size();
		if (k != elts.size()-1)
		  ++ocr_size; // space between words
	      }
	    max_ocr_length = std::max(max_ocr_length,ocr_size);
	  }
	++clines;
      }
    if (train_db)
      {
	_logger->info("ctc/ocr dataset training size={}",clines);
	_db_batchsize = clines;
      }
    else
      {
	_logger->info("ctc/ocr dataset testing size={}",clines);
	_db_testbatchsize = clines;
      }
    _logger->info("ctc output string max size={}",max_ocr_length);
    train_file.clear();
    train_file.seekg(0, std::ios::beg);
    
    int cn = (_bw ? 1 : 3);
    int max_lines = std::pow(10,9) / (_height*_width*3*4);
    _logger->info("hdf5 using max number of lines={}",max_lines);
        
    cv::Size size(_width,_height);
    int chunks = std::ceil(clines / static_cast<double>(max_lines));
    _logger->info("proceeding with {} hdf5 chunks",chunks);
    std::vector<std::string> dbchunks;
    for (int ch=0;ch<chunks;ch++)
      {
	int tlines = (ch == chunks-1) ? clines % max_lines: max_lines;
	if (tlines == 0)
	  break;
	boost::multi_array<float,4> img_data(boost::extents[tlines][cn][_height][_width]);
	boost::multi_array<float,2> ocr_data(boost::extents[tlines][max_ocr_length]);
	int nline = 0;
	while (std::getline(train_file, line))
	  {
	    std::vector<std::string> elts = dd_utils::split(line,' ');
	    
	    // first elt is the image path
	    std::string img_path = elts.at(0);
	    cv::Mat img = cv::imread(img_path, _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED :
                               (_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR));
	    if (img.empty())
	      throw InputConnectorBadParamException("failed to read OCR image " + img_path);
	    if (_align && img.rows > img.cols) // rotate so that width is longest axis
	      {
		cv::Mat timg;
		cv::transpose(img,timg);
		cv::flip(timg,img,1);
	      }
	    cv::Mat rimg;
	    try
	      {
		cv::resize(img,rimg,size,0,0,CV_INTER_CUBIC);
	      }
	    catch(std::exception &e)
	      {
		_logger->error("failed resizing image {}: {}",img_path,e.what());
		continue;
	      }
	    img = rimg;
	    for(int r=0;r<img.rows;r++)
	      {
		for(int c=0;c<img.cols;c++)
		  {
		    cv::Point3_<uint8_t> pixel = img.at<cv::Point3_<uint8_t>>(r,c);
		    img_data[nline][0][r][c] = pixel.x; // B
		    img_data[nline][1][r][c] = pixel.y; // G
		    img_data[nline][2][r][c] = pixel.z; // R
		  }
	      }
	    
	    // then come the CTC/OCR string
	    int cpos = 0;
	    for (size_t i=1;i<elts.size();i++)
	      {
		std::string ostr = elts.at(i);
		char *ostr_c = (char*)ostr.c_str();
		char *ostr_ci = ostr_c;
		char *end = ostr_c + strlen(ostr_c);
		while(ostr_ci<end && cpos < max_ocr_length)
		{
		    // check / add / get id from alphabet
		    uint32_t c = utf8::next(ostr_ci,end);
		    int nc = -1;
		    if ((ait=alphabet.find(c))==alphabet.end())
		      {
			if (train_db)
			  {
			    nc = alphabet.size();
			    alphabet.insert(std::pair<uint32_t,int>(c,nc));
			  }
			else
			  {
			    _logger->warn("character {} in test set not found in training set",c);
			    nc = 0; // space, blank
			  }
		      }
		    else nc = (*ait).second;
		    ocr_data[nline][cpos] = static_cast<float>(nc);
		    ++cpos;
		}
		// add space, only if more forthcoming words
		if (cpos < max_ocr_length && i != elts.size()-1)
		  {
		    ocr_data[nline][cpos] = 0.0;
		    ++cpos;
		  }
	      }
	    // complete string with blank label
	    while (cpos < max_ocr_length)
	      {
		ocr_data[nline][cpos] = 0.0;
		++cpos;
	      }
	    if (nline == tlines-1)
	      break;
	    ++nline;
	  }

	std::string dbchunkname = dbfullname + "_" + std::to_string(ch) + ".h5";
	dbchunks.push_back(dbchunkname);
	H5::H5File hdffile(dbchunkname, H5F_ACC_TRUNC);
	std::string dname;
	if (train_db)
	  dname = "train";
	else dname = "test";
	_logger->info("created hdf5 {} dataset for chunk {}",dname,ch);
	
	// create datasets
	// image data
	hsize_t img_dims[4]; 
	img_dims[0] = tlines;
	img_dims[1] = cn;
	img_dims[2] = _height;
	img_dims[3] = _width;
	H5::DataSpace dataspace(4, img_dims);
	H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
	//datatype.setOrder( H5T_ORDER_LE );
	H5::DataSet dataset = hdffile.createDataSet("data",datatype,dataspace);
	dataset.write(img_data.data(), H5::PredType::NATIVE_FLOAT);
	
	// ocr data
	hsize_t ocr_dims[2]; 
	ocr_dims[0] = tlines;
	ocr_dims[1] = max_ocr_length;
	H5::DataSpace dataspace2(2, ocr_dims);
	H5::FloatType datatype2(H5::PredType::NATIVE_FLOAT);
	//datatype.setOrder( H5T_ORDER_LE );
	H5::DataSet dataset2 = hdffile.createDataSet("label",datatype2,dataspace2);
	dataset2.write(ocr_data.data(), H5::PredType::NATIVE_FLOAT);
      }
    
    // generate list of hdf5 db files
    std::ofstream tlist(dblistfilename.c_str());
    for (auto s: dbchunks)
      tlist << s << std::endl;
    tlist.close();
  }
	#endif // USE_HDF5

  int ImgCaffeInputFileConn::objects_to_db(const std::vector<std::string> &filelists,
					   const int &db_height,
					   const int &db_width,
					   const std::string &traindbname,
					   const std::string &testdbname,
					   const bool &encoded,
					   const std::string &encode_type,
					   const std::string &backend)
  {
    // bypass creation if dbs already exist
    if (fileops::file_exists(traindbname))
      {
	std::string line;
	std::ifstream fin_mean(_model_repo + "/mean_values.txt");
	_mean_values.clear();
	while(std::getline(fin_mean,line))
	  {
	    std::vector<std::string> elts = dd_utils::split(line,' ');
	    for (auto s: elts)
	      _mean_values.push_back(std::atof(s.c_str()));
	    break;
	  }

	_logger->warn("object db file {} already exists, bypassing creation",traindbname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(traindbname.c_str(), db::READ);
	_db_batchsize = db->Count();
	_logger->warn("image db file {} with {} records",traindbname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbname))
	  {
	    _logger->warn("image db file {} already exists, bypassing creation but checking on records",testdbname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->warn("image db file {} with {} records",testdbname,_db_testbatchsize);
	  }
	return 0;
      }

    // read train lines
    std::ifstream in(filelists.at(0));
    if (!in.is_open())
      throw InputConnectorBadParamException("failed opening training data file " + filelists.at(0));
    _logger->info("reading training data file {}",filelists.at(0));
    std::string line;
    std::vector<std::pair<std::string,std::string>> lines;
    int clines = 0;
    while(std::getline(in,line))
      {
	std::vector<std::string> elts = dd_utils::split(line,' ');
	if (elts.size() != 2)
	  throw InputConnectorBadParamException("wrong line " + std::to_string(clines) + " in data file " + filelists.at(0));
	lines.push_back(std::pair<std::string,std::string>(elts.at(0),elts.at(1)));
	++clines;
      }
    
    //TODO: shuffle & split
    
    // create train db
    write_objects_to_db(traindbname,db_height,db_width,lines,encoded,encode_type,backend,true);
    _db_batchsize = lines.size();
    
    // read test lines as needed
    if (filelists.size() < 2)
      return 0;
    std::ifstream tin(filelists.at(1));
    if (!tin.is_open())
      throw InputConnectorBadParamException("failed opening testing data file " + filelists.at(1));
    _logger->info("reading testing data file {}",filelists.at(1));
    std::vector<std::pair<std::string,std::string>> tlines;
    clines = 0;
    while(std::getline(tin,line))
      {
	std::vector<std::string> elts = dd_utils::split(line,' ');
	if (elts.size() != 2)
	  throw InputConnectorBadParamException("wrong line " + std::to_string(clines) + " in data file " + filelists.at(1));
	tlines.push_back(std::pair<std::string,std::string>(elts.at(0),elts.at(1)));
	++clines;
      }
    write_objects_to_db(testdbname,this->_height,this->_width,tlines,encoded,encode_type,backend,false);
    _db_testbatchsize = tlines.size();
    
    //TODO: write corresp / map file
    
    return 0;
  }

  void ImgCaffeInputFileConn::write_objects_to_db(const std::string &dbfullname,
						  const int &db_height,
						  const int &db_width,
						  const std::vector<std::pair<std::string,std::string>> &lines,
						  const bool &encoded,
						  const std::string &encode_type,
						  const std::string &backend,
						  const bool &train)
  {

    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    AnnotatedDatum_AnnotationType type = AnnotatedDatum_AnnotationType_BBOX; 

    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;
    int min_dim = 0;
    int max_dim = 0;
    std::string label_type = "txt";
    bool status = true;
    bool check_size = false; // check whether all datum have the same size
    std::vector<float> meanv;
    
    std::map<std::string, int> name_to_label;
    std::string enc = encode_type;
    
    for (size_t line_id = 0; line_id < lines.size(); ++line_id)
      {
	AnnotatedDatum anno_datum;
	Datum* datum = anno_datum.mutable_datum();
	if (encoded && !enc.size())
	  {
	    // Guess the encoding type from the file name
	    string fn = lines[line_id].first;
	    size_t p = fn.rfind('.');
	    if ( p == fn.npos )
	      _logger->warn("failed to guess the encoding of '{}",fn);
	    enc = fn.substr(p);
	    std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
	    _logger->info("using encoding {}",enc);
	  }
	std::string filename = lines[line_id].first;
	std::string labelname = lines[line_id].second;
	int width = db_width; // do not resize images is default
	int height = db_height;
	status = ReadRichImageToAnnotatedDatum(filename, labelname, height,
					       width, min_dim, max_dim, !_bw, enc, type, label_type,
					       name_to_label, &anno_datum);
	anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
	if (status == false)
	  {
	    _logger->error("failed to read {} or {}",lines[line_id].first,lines[line_id].second);
	    throw InputConnectorBadParamException("failed to read " + lines[line_id].first + " or " + lines[line_id].second + " at line " + std::to_string(line_id));
	  }
	if (check_size)
	  {
	    if (!data_size_initialized)
	      {
		data_size = datum->channels() * datum->height() * datum->width();
		data_size_initialized = true;
	      }
	    else
	      {
		const std::string& data = datum->data();
	        if (static_cast<int>(data.size()) != data_size)
		  {
		    _logger->error("incorrect data field size {}",data.size());
		    throw InputConnectorBadParamException("incorrect data field size " + std::to_string(data.size()));
		  }
	      }
	  }

	// compute the mean
	if (train)
	  {
	    if (_mean_values.empty())
	      _mean_values = std::vector<float>(datum->channels(),0.0);
	    std::vector<float> lmeanv(datum->channels(),0.0);
	    std::vector<cv::Mat> channels;
	    cv::Mat img = cv::imread(lines[line_id].first);
	    cv::split(img, channels);
	    for (int d=0;d<datum->channels();d++) {
	      lmeanv[d] = cv::mean(channels[d])[0];
	      _mean_values[d] += lmeanv[d];
	    }
	  }
	
	// sequential
	string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;
	
	// Put in db
	string out;
	CHECK(anno_datum.SerializeToString(&out));
	if (!anno_datum.SerializeToString(&out))
	  _logger->error("Failed serialization of annotated datum for db storage");
	txn->Put(key_str, out);
	
	if (++count % 1000 == 0) {
	  // Commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  LOG(INFO) << "Processed " << count << " files.";
	}
      }
  
    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << count << " files.";
    }

    // average the mean
    if (train)
      {
	std::ofstream fout_mean(_model_repo + "/mean_values.txt"); // since images are of various sizes
	for (size_t i=0;i<_mean_values.size();i++)
	  {
	    _mean_values.at(i) /= static_cast<float>(count);
	    fout_mean << _mean_values.at(i) << " ";
	  }
	fout_mean << std::endl;
	fout_mean.close();
      }
  }
  
  int ImgCaffeInputFileConn::compute_images_mean(const std::string &dbname,
						 const std::string &meanfile,
						 const std::string &backend)
  {
    std::string dbfullname = dbname + "." + backend;
    if (fileops::file_exists(meanfile))
      {
	_logger->warn("image mean file {} already exists, bypassing creation",meanfile);
	BlobProto sum_blob;
	ReadProtoFromBinaryFile(meanfile.c_str(),&sum_blob);
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	_mean_values = std::vector<float>(channels,0.0);
	_logger->info("Number of channels: {}",channels);
	for (int c = 0; c < channels; ++c) {
	  for (int i = 0; i < dim; ++i) {
	    _mean_values[c] += sum_blob.data(dim * c + i);
	  }
	  _logger->info("mean value channel [{}]:{}",c,_mean_values[c] / dim);
	  _mean_values[c] /= dim;
	}
	return 0;
      }

    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::READ);
    std::unique_ptr<db::Cursor> cursor(db->NewCursor());

    BlobProto sum_blob;
    int count = 0;
    // load first datum
    Datum datum;
    datum.ParseFromString(cursor->value());

    if (DecodeDatumNative(&datum)) {
      //_logger->info("Decoding Datum");
    }
    
    sum_blob.set_num(1);
    sum_blob.set_channels(datum.channels());
    sum_blob.set_height(datum.height());
    sum_blob.set_width(datum.width());
    int size_in_datum = std::max<int>(datum.data().size(),
				      datum.float_data_size());
    for (int i = 0; i < size_in_datum; ++i) {
      sum_blob.add_data(0.);
    }
    while (cursor->valid()) {
      Datum datum;
      datum.ParseFromString(cursor->value());
      DecodeDatumNative(&datum);
      
      const std::string& data = datum.data();
      size_in_datum = std::max<int>(datum.data().size(),
				    datum.float_data_size());
      if (data.size() != 0) {
	for (int i = 0; i < size_in_datum; ++i) {
	  sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
	}
      } else {
	for (int i = 0; i < size_in_datum; ++i) {
	  sum_blob.set_data(i, sum_blob.data(i) +
			    static_cast<float>(datum.float_data(i)));
	}
      }
      ++count;
      if (count % 10000 == 0) {
	_logger->info("Processed {} files",count);
      }
      cursor->Next();
    }
    
    if (count % 10000 != 0) {
      _logger->info("Processed {} files",count);
    }
    for (int i = 0; i < sum_blob.data_size(); ++i) {
      sum_blob.set_data(i, sum_blob.data(i) / count);
    }
    // Write to disk
    _logger->info("Write to {}",meanfile);
    WriteProtoToBinaryFile(sum_blob, meanfile.c_str());

    // let's store the simpler mean values in case of image
    // size changes, e.g. cropping
    const int channels = sum_blob.channels();
    const int dim = sum_blob.height() * sum_blob.width();
    _mean_values = std::vector<float>(channels,0.0);
    _logger->info("Number of channels: {}",channels);
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < dim; ++i) {
	_mean_values[c] += sum_blob.data(dim * c + i);
      }
      _logger->info("mean value channel [{}]:{}",c,_mean_values[c] / dim);
      _mean_values[c] /= dim;
    }
    return 0;
  }

  std::vector<caffe::Datum> ImgCaffeInputFileConn::get_dv_test_db(const int &num,
							       const bool &has_mean_file)
  {
    static Blob<float> data_mean;
    static float *mean = nullptr;
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
	
	// open mean file if any
	std::string meanfullname = _model_repo + "/" + _meanfname;
	if (has_mean_file && fileops::file_exists(meanfullname))
	  {
	    BlobProto blob_proto;
	    ReadProtoFromBinaryFile(meanfullname.c_str(),&blob_proto);
	    data_mean.FromProto(blob_proto);
	    mean = data_mean.mutable_cpu_data();
	  }
      }
    std::vector<caffe::Datum> dv;
    int i =0;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == tnum)
	  break;
	Datum datum;
	datum.ParseFromString(_test_db_cursor->value());
	std::vector<double> fd; // XXX: hack to work around removal of float_data in decoder
	for (int s=0;s<datum.float_data_size();s++)
	  fd.push_back(datum.float_data(s));
	DecodeDatumNative(&datum);
	for (auto s: fd)
	  datum.add_float_data(s);

	// deal with the mean image values, this forces to turn the datum
	// data into an array of floats (as opposed to original bytes format)
	// XXX: beware this is not compatible with imagedata layer with multi-labels as they are stored in float_data
	if (mean)
	  {
	    int height = datum.height();
	    int width = datum.width();
	    for (int c=0;c<datum.channels();++c)
	      for (int h=0;h<height;++h)
		for (int w=0;w<width;++w)
		  {
		    int data_index = (c*height+h)*width+w;
		    float datum_element;
		    datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
		    datum.add_float_data(datum_element - mean[data_index]);
		  }
	    datum.clear_data();
	  }
	dv.push_back(datum);
	_ids.push_back(_test_db_cursor->key());
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }

  std::vector<caffe::Datum> ImgCaffeInputFileConn::get_dv_test_segmentation(const int &num,
									    const bool &has_mean_file)
  {
    (void) has_mean_file;
    if (_segmentation_data_lines.empty())
      {
	_logger->info("reading segmentation test file {}",_uris.at(1).c_str());
	std::ifstream infile(_uris.at(1).c_str());
	std::string filename, label_filename;
	while(infile >> filename >> label_filename)
	  {
	    _segmentation_data_lines.push_back(std::make_pair(filename,label_filename));
	  }
      }

    std::vector<caffe::Datum> dv;
    int j = 0;
    for (int i=_dt_seg;i<static_cast<int>(_segmentation_data_lines.size());i++)
      {
	if (j == num)
	  break;
	std::string enc = guess_encoding(_segmentation_data_lines[i].first);
	Datum datum_data, datum_labels;
	bool status = ReadImageToDatum(_segmentation_data_lines[i].first,-1,
				       _height,_width,0,0,!_bw,false,enc,&datum_data);
	if (status == false)
	  {
	    _logger->error("reading segmentation image {} to datum",_segmentation_data_lines[i].first);
	    continue;
	  }

	cv::Mat cv_lab = ReadImageToCVMat(_segmentation_data_lines[i].second,_height,_width,false,true);
	datum_labels.set_height(_height);
	datum_labels.set_width(_width);
	datum_labels.set_channels(1);
	for (int j=0;j<cv_lab.rows;j++) // height
	  {
	    for (int i=0;i<cv_lab.cols;i++) // width
	      {
		datum_labels.add_float_data(static_cast<float>(cv_lab.at<uchar>(j,i)));
	      }
	  }
	dv.push_back(datum_data);
	dv.push_back(datum_labels);
	_ids.push_back(std::to_string(j));
	++j;
      }
    _dt_seg += j;
    return dv;
  }
  
  void ImgCaffeInputFileConn::reset_dv_test()
  {
    _dt_vit = _dv_test.begin();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db = std::unique_ptr<caffe::db::DB>();
    _dt_seg = 0;
  }


  /*- DDCCsv -*/
  int DDCCsv::read_file(const std::string &fname)
  {
    if (_cifc)
      {
        _cifc->read_csv(fname);
        return 0;
      }
    else return -1;
  }

  int DDCCsv::read_db(const std::string &fname)
  {
    _cifc->_db_fname = fname;
    return 0;
  }
  
  int DDCCsv::read_mem(const std::string &content)
  {
    if (!_cifc)
      return -1;
    std::vector<double> vals;
    std::string cid;
    int nlines = 0;
    _cifc->read_csv_line(content,_cifc->_delim,vals,cid,nlines);
    if (_cifc->_scale)
      _cifc->scale_vals(vals);
    if (!cid.empty())
      _cifc->_csvdata.emplace_back(cid,vals);
    else _cifc->_csvdata.emplace_back(std::to_string(_cifc->_csvdata.size()+1),vals);
    return 0;
  }

  /*- CSVCaffeInputFileConn -*/
  int CSVCaffeInputFileConn::csv_to_db(const std::string &traindbname,
				       const std::string &testdbname,
				       const APIData &ad_input,
				       const std::string &backend)
  {
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;

    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    if (fileops::file_exists(dbfullname))
      {
	_logger->warn("CSV db file {} already exists, bypassing creation but checking on records",dbfullname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	while(cursor->valid())
	  {
	    if (_channels == 0)
	      {
		Datum datum;
		datum.ParseFromString(cursor->value());
		_channels = datum.channels();
	      }
	    break;
	  }
	_db_batchsize = db->Count();
	_logger->info("CSV db train file {} with {} records",dbfullname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    _logger->warn("CSV db file {} already exists, bypassing creation but checking on records",testdbfullname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->info("CSV db test file {} with {} records",testdbfullname,_db_testbatchsize);
	  }	
	return 0;
      }
    
    // write files to dbs (i.e. train and possibly test)
    _db_batchsize = 0;
    _db_testbatchsize = 0;
    write_csvline_to_db(dbfullname,testdbfullname,ad_input);
        
    return 0;
  }

  void CSVCaffeInputFileConn::add_train_csvline(const std::string &id,
						std::vector<double> &vals)
  {
    if (!_db)
      {
	CSVInputFileConn::add_train_csvline(id,vals);
	return;
      }

    static int count = 0;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    Datum d = to_datum(vals);
    
    // sequential
    int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(count).c_str()); // XXX: using appeared to confuse the training (maybe because sorted)
    
    // put in db
    std::string out;
    if(!d.SerializeToString(&out))
      {
	_logger->error("Failed serialization of datum for db storage");
	return;
      }
    if (!_txn)
      {
	_logger->error("db transaction cannot be executed, wrong db flag ?");
	throw InputConnectorBadParamException("db transaction cannot be executed, wrong db flag ?");
      }
    _txn->Put(std::string(key_cstr, length), out);
    _db_batchsize++;
    
    if (++count % 10000 == 0) {
      // commit db
      _txn->Commit();
      _txn.reset(_tdb->NewTransaction());
      _logger->info("Processed {} records",count);
    }
  }

  void CSVCaffeInputFileConn::add_test_csvline(const std::string &id,
					       std::vector<double> &vals)
  {
    if (!_db)
      {
	CSVInputFileConn::add_test_csvline(id,vals);
	return;
      }
    
      static int count = 0;
    
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    Datum d = to_datum(vals);
    
    // sequential
    int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(count).c_str()); // XXX: using id appeared to confuse the training (maybe because sorted)
    
    // put in db
    std::string out;
    if(!d.SerializeToString(&out))
      {
	_logger->error("Failed serialization of datum for db storage");
	return;
      }
    _ttxn->Put(std::string(key_cstr, length), out);
    _db_testbatchsize++;

    if (++count % 10000 == 0) {
      // commit db
      _ttxn->Commit();
      _ttxn.reset(_ttdb->NewTransaction());
      _logger->info("Processed {} records",count);
    }
  }
  
  void CSVCaffeInputFileConn::write_csvline_to_db(const std::string &dbfullname,
						  const std::string &testdbfullname,
						  const APIData &ad_input,
						  const std::string &backend)
  {
    // Create new DB
    _tdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _tdb->Open(dbfullname.c_str(), db::NEW);
    _txn = std::unique_ptr<db::Transaction>(_tdb->NewTransaction());
    _ttdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _ttdb->Open(testdbfullname.c_str(), db::NEW);
    _ttxn = std::unique_ptr<db::Transaction>(_ttdb->NewTransaction());

    _csv_fname = _uris.at(0); // training only from file
    if (!fileops::file_exists(_csv_fname))
      throw InputConnectorBadParamException("training CSV file " + _csv_fname + " does not exist");
    if (_uris.size() > 1)
      _csv_test_fname = _uris.at(1);
    /*if (ad_input.has("label"))
      _label = ad_input.get("label").get<std::string>();
    else if (_train && _label.empty()) throw InputConnectorBadParamException("missing label column parameter");
    if (ad_input.has("label_offset"))
    _label_offset = ad_input.get("label_offset").get<int>();*/
    
    DataEl<DDCCsv> ddcsv;
    ddcsv._ctype._cifc = this;
    ddcsv._ctype._adconf = ad_input;
    ddcsv.read_element(_csv_fname,this->_logger);

    _txn->Commit();
    _ttxn->Commit();
    
    _tdb->Close();
    _ttdb->Close();
  }

  std::vector<caffe::Datum> CSVCaffeInputFileConn::get_dv_test_db(const int &num)
  {
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    std::vector<caffe::Datum> dv;
    int i =0;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == tnum)
	  break;
	Datum datum;
	datum.ParseFromString(_test_db_cursor->value());
	dv.push_back(datum);
	_ids.push_back(_test_db_cursor->key());
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }

  void CSVCaffeInputFileConn::reset_dv_test()
  {
    _dt_vit = _dv_test.begin();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db = std::unique_ptr<caffe::db::DB>();
  }

  /*- TxtCaffeInputFileConn -*/
  int TxtCaffeInputFileConn::txt_to_db(const std::string &traindbname,
				       const std::string &testdbname,
				       const std::string &backend)
  {
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;

    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    if (fileops::file_exists(dbfullname))
      {
	_logger->warn("Txt db file {} already exists, bypassing creation but checking on records",dbfullname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	while(cursor->valid())
	  {
	    if (_channels == 0)
	      {
		Datum datum;
		datum.ParseFromString(cursor->value());
		_channels = datum.channels();
	      }
	    break;
	  }
	_db_batchsize = db->Count();
	_logger->info("Txt db train file {} with {} records",dbfullname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    _logger->warn("Txt db file {} already exists, bypassing creation but checking on records",testdbfullname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->info("Txt db test file {} with {} records",testdbfullname,_db_testbatchsize);
	  }
	// XXX: remove in-memory data, which pre-processing is useless and should be avoided
	destroy_txt_entries(_txt);
	destroy_txt_entries(_test_txt);
	
	return 0;
      }
    
    _db_batchsize = _txt.size();
    _db_testbatchsize = _test_txt.size();

    _logger->info("db_batchsize={} / db_testbatchsize={}",_db_batchsize,_db_testbatchsize);
    
    // write to dbs (i.e. train and possibly test)
    if (!_sparse)
      write_txt_to_db(dbfullname,_txt);
    else write_sparse_txt_to_db(dbfullname,_txt);
    destroy_txt_entries(_txt);
    if (!_test_txt.empty())
      {
	if (!_sparse)
	  write_txt_to_db(testdbfullname,_test_txt);
	else write_sparse_txt_to_db(testdbfullname,_test_txt);
	destroy_txt_entries(_test_txt);
      }
    
    return 0;
  }




  int DDCCsvTS::read_file(const std::string &fname, bool is_test_data)
  {
    if (_cifc)
      {
        std::string test_file = _cifc->_csv_test_fname;
        _cifc->_csv_test_fname = "";
        _cifc->read_csv(fname);
        _cifc->push_csv_to_csvts(is_test_data);
        _cifc->_csv_test_fname = test_file;
        _cifc->_ids.push_back(fname);
        return 0;
      }
    else return -1;

  }

  int DDCCsvTS::read_db(const std::string &fname)
  {
    _cifc->_db_fname = fname;
    return 0;
  }

  int DDCCsvTS::read_mem(const std::string &content)
  {

    if (!_cifc)
      return -1;
    _ddcsvts._cifc = _cifc;
    _ddcsvts.read_mem(content);
    return 0;
  }

  int DDCCsvTS::read_dir(const std::string &dir, bool is_test_data)
  {
    // first recursive list csv files
    std::unordered_set<std::string> allfiles;
    int ret = fileops::list_directory(dir, true, false, true, allfiles);
    if (ret != 0)
      return ret;
    // then simply read them
    if (!_cifc)
      return -1;

    if (_cifc->_scale && (_cifc->_min_vals.empty() || _cifc->_max_vals.empty() ))
      {
        std::vector<double> min_vals = _cifc->_min_vals;
        std::vector<double> max_vals = _cifc->_max_vals;
        for (auto fname : allfiles)
          {
            std::pair<std::vector<double>,std::vector<double>> mm = _cifc->get_min_max_vals(fname);
            if (min_vals.empty())
              min_vals = mm.first;
            else
              for (size_t j=0;j<mm.first.size();j++)
                min_vals.at(j) = std::min(mm.first.at(j),min_vals.at(j));
            if (max_vals.empty())
              max_vals = mm.second;
            else
              for (size_t j=0;j<mm.first.size();j++)
                max_vals.at(j) = std::max(mm.second.at(j),max_vals.at(j));
          }
        _cifc->_min_vals = min_vals;
        _cifc->_max_vals = max_vals;
        _cifc->serialize_bounds();
      }

    if (!is_test_data && _cifc->_shuffle)
      {
        std::vector<std::string> allfiles_v;
        for (auto fname : allfiles)
          allfiles_v.push_back(fname);
        auto rng = std::default_random_engine();
        std::shuffle(allfiles_v.begin(), allfiles_v.end(), rng);
        for (auto fname : allfiles_v)
          read_file(fname, is_test_data);
      }
    else

      for (auto fname : allfiles)
        read_file(fname, is_test_data);

    return 0;
}

  void CSVTSCaffeInputFileConn::set_datadim(bool is_test_data)
  {
    if (_datadim != -1)
      return;
    if (is_test_data)
      _datadim = _csvtsdata_test[0][0]._v.size() +1;
    else
      _datadim = _csvtsdata[0][0]._v.size() +1;
  }

  void CSVTSCaffeInputFileConn::push_csv_to_csvts(bool is_test_data)
  {

    CSVTSInputFileConn::push_csv_to_csvts(is_test_data);
    set_datadim(is_test_data);
    dv_to_db(is_test_data);
  }

  void CSVTSCaffeInputFileConn::transform(const APIData &ad)
  {// calls CSVTSInputfileconn::transform and db stuff (in particular cvsts_to_db)
    APIData ad_param = ad.getobj("parameters");
    APIData ad_input = ad_param.getobj("input");
    APIData ad_mllib = ad_param.getobj("mllib");

    get_data(ad);
    fillup_parameters(ad_input);

    if (_train && ad_input.has("db") && ad_input.get("db").get<bool>())
      {
        _dbfullname = _model_repo + "/" + _dbfullname;
        _test_dbfullname = _model_repo + "/" + _test_dbfullname;

        _db = true;
        csvts_to_db(_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname,
		    ad_input);
        write_class_weights(_model_repo,ad_mllib);

        // enrich data object with db files location
        APIData dbad;
        dbad.add("train_db",_dbfullname);
        if (_test_split > 0.0)
          dbad.add("test_db",_test_dbfullname);
        const_cast<APIData&>(ad).add("db",dbad);
	}
      else
	{
	  try
	    {
	      CSVTSInputFileConn::transform(ad);
             set_datadim();
             this->_ids=this->_fnames;
	    }
	  catch (std::exception &e)
	    {
	      throw;
	    }

	  // transform to datum by filling up float_data
	  if (_train)
	    {
             csvts_to_dv(false,true,true,true,false);
	    }
	  if (!_train)
	    {
	      if (!_db_fname.empty())
		{
		  _test_dbfullname = _db_fname;
		  _db = true;
		  return; // done
		}
	      _csvtsdata_test = std::move(_csvtsdata);
	    }
	  else _csvtsdata.clear();
         csvts_to_dv(true,true,true,false,_continuation);
	  _csvtsdata_test.clear();
	}
      _csvtsdata_test.clear();
  }

  void CSVTSCaffeInputFileConn::reset_dv_test()
  {
    _dt_vit = _dv_test.begin();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db  = std::unique_ptr<caffe::db::DB>();
  }


  std::vector<caffe::Datum> CSVTSCaffeInputFileConn::get_dv_test(const int &num,
                                                                 const bool &has_mean_file)
  {
    (void)has_mean_file;
    if (!_db)
      {
        int i = 0;
        std::vector<caffe::Datum> dv;
        while(_dt_vit!=_dv_test.end()
              && i < num)
          {
            dv.push_back((*_dt_vit));
            ++i;
            ++_dt_vit;
          }
        return dv;
      }
    else return get_dv_test_db(num);
  }

  std::vector<caffe::Datum> CSVTSCaffeInputFileConn::get_dv_test_db(const int &num)
  {
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
        // open db and create cursor
        if (!_test_db)
          {
            _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
            _test_db->Open(_test_dbfullname.c_str(),db::READ);
          }
        _test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    std::vector<caffe::Datum> dv;
    int i =0;
    while(_test_db_cursor->valid())
      {
        // fill up a vector up to 'num' elements.
        if (i == tnum)
          break;
        Datum datum;
        datum.ParseFromString(_test_db_cursor->value());
        dv.push_back(datum);
        _ids.push_back(_test_db_cursor->key());
        _test_db_cursor->Next();
        ++i;
      }
    return dv;
  }



  int CSVTSCaffeInputFileConn::csvts_to_db(const std::string &traindbname,
                                           const std::string &testdbname,
                                           const APIData &ad_input,
                                           const std::string &backend) // lmdb, leveldb
  {
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;

    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    if (fileops::file_exists(dbfullname))
      {
	_logger->warn("CSV db file {} already exists, bypassing creation but checking on records",dbfullname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	while(cursor->valid())
	  {
	    if (_channels == 0 || _datadim == -1)
	      {
		Datum datum;
		datum.ParseFromString(cursor->value());
		_channels = datum.channels();
              _datadim = datum.height();
	      }
	    break;
	  }
	_db_batchsize = db->Count();
	_logger->info("CSV db train file {} with {} records",dbfullname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    _logger->warn("CSV db file {} already exists, bypassing creation but checking on records",testdbfullname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->info("CSV db test file {} with {} records",testdbfullname,_db_testbatchsize);
	  }
	return 0;
      }
    // write files to dbs (i.e. train and possibly test)
    _db_batchsize = 0;
    _db_testbatchsize = 0;
    write_csvts_to_db(dbfullname, testdbfullname, ad_input,backend);
    return 0;
  }

  void CSVTSCaffeInputFileConn::dv_to_db(bool is_test_data)
  {
    if (!_db)
      return;
    std::vector<Datum>& dvtodump = _dv;
    int &dv_index = _dv_index;
    std::unique_ptr<caffe::db::Transaction> *txn_ptr = &_txn;
    std::unique_ptr<caffe::db::DB> *tdb_ptr = &_tdb;

    csvts_to_dv(is_test_data,true,true,true,_continuation);
    if (is_test_data)
      {
        dvtodump = _dv_test;
        dv_index = _dv_test_index;
        txn_ptr = &_ttxn;
        tdb_ptr = &_ttdb;
      }

    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    int count = 0;

    for (Datum d:dvtodump)
      {
        std::string out;

        int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(count).c_str()); // XXX: using appeared to confuse the training (maybe because sorted)
        if(!d.SerializeToString(&out))
          {
            _logger->error("Failed serialization of datum for db storage");
            return;
          }
        if (!*txn_ptr)
          {
            _logger->error("db transaction cannot be executed, wrong db flag ?");
            throw InputConnectorBadParamException("db transaction cannot be executed, wrong db flag ?");
          }
        (*txn_ptr)->Put(std::string(key_cstr, length), out);
        _db_batchsize++;

        if (++count % 10 == 0) {
          // commit db
          (*txn_ptr)->Commit();
          (*txn_ptr).reset((*tdb_ptr)->NewTransaction());
          _logger->info("Processed {} timeseries",count);
        }
      }
    dvtodump.clear();
    dv_index = -1;
  }

  void CSVTSCaffeInputFileConn::write_csvts_to_db(const std::string &dbfullname,
                                                  const std::string &testdbfullname,
                                                  const APIData &ad_input,
                                                  const std::string &backend)
  {
    // Create new DB
    _tdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _tdb->Open(dbfullname.c_str(), db::NEW);
    _txn = std::unique_ptr<db::Transaction>(_tdb->NewTransaction());
    _ttdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _ttdb->Open(testdbfullname.c_str(), db::NEW);
    _ttxn = std::unique_ptr<db::Transaction>(_ttdb->NewTransaction());
    _csv_fname = _uris.at(0); // training only from file

    if (!fileops::dir_exists(_csv_fname))
      throw InputConnectorBadParamException("training CSV_TS dir " + _csv_fname + " does not exist");
    if (_uris.size() > 1)
      _csv_test_fname = _uris.at(1);
    DDCCsvTS ddccsvts;
    ddccsvts._cifc = this;
    ddccsvts._adconf = ad_input;
    ddccsvts.read_dir(_csv_fname);
    ddccsvts.read_dir(_csv_test_fname,true);


    _txn->Commit();
    _ttxn->Commit();

    _tdb->Close();
    _ttdb->Close();

  }


  void CSVTSCaffeInputFileConn::csvts_to_dv(bool test, bool clear_dv_first, bool clear_csvts_after, bool split_seqs, bool first_is_cont)
  {

    std::vector<Datum> * dv;
    std::vector<std::vector<CSVline>> * data;
    int* index;
    if (test)
      {
        dv = & _dv_test;
        index = &_dv_test_index;
        data = &this->_csvtsdata_test;
      }
    else
      {
        dv = &_dv;
        index = &_dv_index;
        data = &this->_csvtsdata;
      }

    if (clear_dv_first)
      {
        dv->clear();
        *index = -1;
      }

    Datum *d = NULL;

    if (*index == -1 || *index == _timesteps)
      {
        Datum dprim;
        dprim.set_channels(this->_timesteps);
        dprim.set_height(this->_datadim);
        dprim.set_width(1);
        for (int i=0; i< this->_timesteps*this->_datadim; ++i)
          dprim.add_float_data(0.0);
        dv->push_back(dprim);
        d = &(dv->back());
        *index = 0;
      }
    else
      d = &(dv->back());

    for (unsigned int si=0; si< data->size(); ++si)
      {
        unsigned int offset = 0;
        unsigned int ti = 0;
        while (ti < data->at(si).size())
          {
            if ((ti == 0 && !first_is_cont) || ((*index == 0) && split_seqs))
              d->set_float_data(*index*this->_datadim,0.0); // new sequence
            else
              d->set_float_data(*index*this->_datadim,1.0); // continue sequence
            for (unsigned int di =0; di < _label_pos.size(); ++di)
                d->set_float_data(*index*this->_datadim + di + 1,
                                  data->at(si)[ti]._v[_label_pos[di]]);
            int ii =0;
            for (int di = 0; di<this->_datadim-1; ++di)
              if (std::find(_label_pos.begin(),_label_pos.end(), di) == _label_pos.end())
                  d->set_float_data(*index*this->_datadim + 1 + ii++ + _label_pos.size(),
                                    data->at(si)[ti]._v[di]);
            (*index)++;
            ++ti;
            if ((*index) == _timesteps)
              {
                (*index) = 0;
                if (ti == data->at(si).size())
                  break;
                Datum dprim;
                dprim.set_channels(this->_timesteps);
                dprim.set_height(this->_datadim);
                for (int i=0; i< this->_timesteps*this->_datadim; ++i)
                  dprim.add_float_data(0.0);
                dprim.set_width(1);
                dv->push_back(dprim);
                d = &(dv->back());
                if (ti + _timesteps >= data->at(si).size()+1)
                  {
                    unsigned int tti = data->at(si).size() - _timesteps;
                    for ( ; tti < data->at(si).size(); ++tti)
                      {
                        if (split_seqs)
                          d->set_float_data(*index*this->_datadim,0.0); // continue sequence
                        else
                          d->set_float_data(*index*this->_datadim,1.0); // continue sequence
                        for (unsigned int di =0; di < _label_pos.size(); ++di)
                          d->set_float_data(*index*this->_datadim + di + 1,
                                            data->at(si)[ti]._v[_label_pos[di]]);
                        int ii =0;
                        for (int di = 0; di<this->_datadim-1; ++di)
                          if (std::find(_label_pos.begin(),_label_pos.end(), di) == _label_pos.end())
                            d->set_float_data(*index*this->_datadim + 1 + ii++ + _label_pos.size(),
                                              data->at(si)[ti]._v[di]);
                        (*index)++;
                      }
                    break;
                  }
                offset += _offset;
                ti = offset;
              }
          }

        // at end of sequence start a new datum so that if the sequence
        //is smaller than the number of timesteps,  then every datum contains an independant sequence
        // is sequence is larger than timesteps, it will be splitted into independant sequences
        if (si < data->size() -1)
          {
            Datum dprim;
            dprim.set_channels(this->_timesteps);
            dprim.set_height(this->_datadim);
            for (int i=0; i< this->_timesteps*this->_datadim; ++i)
              dprim.add_float_data(0.0);
            dprim.set_width(1);
            dv->push_back(dprim);
            d = &(dv->back());
            *index = 0;
          }
      }
    if (clear_csvts_after)
        data->clear();

    if (!test && _shuffle)
      {
        _logger->info("shuffling {} datum vectors",_dv.size());
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(_dv.begin(), _dv.end(),g);
      }

  }


  void TxtCaffeInputFileConn::write_txt_to_db(const std::string &dbfullname,
					      std::vector<TxtEntry<double>*> &txt,
					      const std::string &backend)
  {
    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    Datum datum;
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    int n = 0;
    auto hit = txt.begin();
    while(hit!=txt.end())
      {
	if (_characters)
	  datum = to_datum<TxtCharEntry>(static_cast<TxtCharEntry*>((*hit)));
	else datum = to_datum<TxtBowEntry>(static_cast<TxtBowEntry*>((*hit)));
	if (_channels == 0)
	  _channels = datum.channels();
	int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(n).c_str());
	
	// put in db
	std::string out;
	if (!datum.SerializeToString(&out))
	  _logger->error("Failed serialization of datum for db storage");
	txn->Put(string(key_cstr,length),out);

	if (++count % 1000 == 0) {
	  // commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  _logger->info("Processed {} text entries",count);
	}
	
	++hit;
	++n;
      }

    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      _logger->info("Processed {} text entries",count);
    }

    db->Close();
  }

  void TxtCaffeInputFileConn::write_sparse_txt_to_db(const std::string &dbfullname,
						     std::vector<TxtEntry<double>*> &txt,
						     const std::string &backend)
  {
    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    SparseDatum datum;
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    int n = 0;
    auto hit = txt.begin();
    while(hit!=txt.end())
      {
	/*if (_characters)
	  datum = to_datum<TxtCharEntry>(static_cast<TxtCharEntry*>((*hit)));
	  else*/
	datum = to_sparse_datum(static_cast<TxtBowEntry*>((*hit)));
	int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(n).c_str());
	
	// put in db
	std::string out;
	if (!datum.SerializeToString(&out))
	  _logger->error("Failed serialization of datum for db storage");
	txn->Put(string(key_cstr,length),out);

	if (++count % 1000 == 0) {
	  // commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  _logger->info("Processed {} text entries",count);
	}
	
	++hit;
	++n;
      }

    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      _logger->info("Processed {} text entries",count);
    }

    db->Close();
  }

  std::vector<caffe::Datum> TxtCaffeInputFileConn::get_dv_test_db(const int &num)
  {
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    int i =0;
    std::vector<caffe::Datum> dv;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == tnum)
	  break;
	Datum datum;
	datum.ParseFromString(_test_db_cursor->value());
	dv.push_back(datum);
	_ids.push_back(_test_db_cursor->key());
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }
   
  std::vector<caffe::SparseDatum> TxtCaffeInputFileConn::get_dv_test_sparse_db(const int &num)
  {
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    int i =0;
    std::vector<caffe::SparseDatum> dv;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == tnum)
	  break;
	SparseDatum datum;
	datum.ParseFromString(_test_db_cursor->value());
	dv.push_back(datum);
	_ids.push_back(_test_db_cursor->key());
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }

  /*- SVMCaffeInputFileConn -*/
  int SVMCaffeInputFileConn::svm_to_db(const std::string &traindbname,
				       const std::string &testdbname,
				       const APIData &ad_input,
				       const std::string &backend)
  {
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;

    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    if (fileops::file_exists(dbfullname))
      {
	_logger->warn("SVM db file {} already exists, bypassing creation but checking on records",dbfullname);
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	while(cursor->valid())
	  {
	    if (_channels == 0)
	      {
		SparseDatum datum;
		datum.ParseFromString(cursor->value());
		_channels = datum.size();
	      }
	    break;
	  }
	_db_batchsize = db->Count();
	_logger->info("SVM db train file {} with {} records",dbfullname,_db_batchsize);
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {

	    _logger->warn("SVM db file {} already exists, bypassing creation but checking on records",testdbfullname);
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    _logger->info("SVM db test file {} with {} records",testdbfullname,_db_testbatchsize);
	  }	
	return 0;
      }
    
    // write files to dbs (i.e. train and possibly test)
    _db_batchsize = 0;
    _db_testbatchsize = 0;
    write_svmline_to_db(dbfullname,testdbfullname,ad_input);
    
    return 0;
  }

  void SVMCaffeInputFileConn::add_train_svmline(const int &label,
						const std::unordered_map<int,double> &vals,
						const int &count)
  {
    if (!_db)
      {
	SVMInputFileConn::add_train_svmline(label,vals,count);
	return;
      }

    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    SparseDatum d = to_sparse_datum(SVMline(label,vals));
    
    // sequential
    int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(count).c_str()); // XXX: using appeared to confuse the training (maybe because sorted)
    
    // put in db
    std::string out;
    if(!d.SerializeToString(&out))
      {
	_logger->info("Failed serialization of datum for db storage");
	return;
      }
    _txn->Put(std::string(key_cstr, length), out);
    _db_batchsize++;
    
    if (count % 10000 == 0) {
      // commit db
      _txn->Commit();
      _txn.reset(_tdb->NewTransaction());
      _logger->info("Processed {} records",count);
    }
  }

  void SVMCaffeInputFileConn::add_test_svmline(const int &label,
					       const std::unordered_map<int,double> &vals,
					       const int &count)
  {
    if (!_db)
      {
	SVMInputFileConn::add_test_svmline(label,vals,count);
	return;
      }

    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    SparseDatum d = to_sparse_datum(SVMline(label,vals));
    
    // sequential
    int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(count).c_str()); // XXX: using id appeared to confuse the training (maybe because sorted)
    
    // put in db
    std::string out;
    if(!d.SerializeToString(&out))
      {
	_logger->error("Failed serialization of datum for db storage");
	return;
      }
    _ttxn->Put(std::string(key_cstr, length), out);
    _db_testbatchsize++;

    if (count % 10000 == 0) {
      // commit db
      _ttxn->Commit();
      _ttxn.reset(_ttdb->NewTransaction());
      _logger->info("Processed {} records",count);
    }
  }
  
  void SVMCaffeInputFileConn::write_svmline_to_db(const std::string &dbfullname,
						  const std::string &testdbfullname,
						  const APIData &ad_input,
						  const std::string &backend)
  {
    _logger->info("SVM line to db / dbfullname={}",dbfullname);

    // Create new DB
    _tdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _tdb->Open(dbfullname.c_str(), db::NEW);
    _txn = std::unique_ptr<db::Transaction>(_tdb->NewTransaction());
    _ttdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _ttdb->Open(testdbfullname.c_str(), db::NEW);
    _ttxn = std::unique_ptr<db::Transaction>(_ttdb->NewTransaction());
    _logger->info("dbs {} / {} opened",dbfullname,testdbfullname);
    
    _svm_fname = _uris.at(0); // training only from file
    if (!fileops::file_exists(_svm_fname))
      throw InputConnectorBadParamException("training SVM file " + _svm_fname + " does not exist");
    if (_uris.size() > 1)
      _svm_test_fname = _uris.at(1);

    DataEl<DDSvm> ddsvm;
    ddsvm._ctype._cifc = this;
    ddsvm._ctype._adconf = ad_input;
    ddsvm.read_element(_svm_fname,this->_logger);

    _txn->Commit();
    _ttxn->Commit();
    
    _tdb->Close();
    _ttdb->Close();
  }

  std::vector<caffe::SparseDatum> SVMCaffeInputFileConn::get_dv_test_sparse_db(const int &num)
  {
    int tnum = num;
    if (tnum == 0)
      tnum = -1;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    std::vector<caffe::SparseDatum> dv;
    int i =0;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == tnum)
	  break;
	SparseDatum datum;
	datum.ParseFromString(_test_db_cursor->value());
	dv.push_back(datum);
	_ids.push_back(_test_db_cursor->key());
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }

  void SVMCaffeInputFileConn::reset_dv_test()
  {
    _dt_vit = _dv_test_sparse.begin();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db = std::unique_ptr<caffe::db::DB>();
  }

}
