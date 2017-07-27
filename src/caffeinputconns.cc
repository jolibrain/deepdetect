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
#include <memory>
#include <glog/logging.h>

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
	LOG(INFO) << "Write class weights to " << cl_file;
	WriteProtoToBinaryFile(cw_blob,cl_file.c_str());
      }
  }
  
  // convert images into db entries
  // a root folder must contain directories as classes holding image
  // files for each class. The name of the class is the name of the directory.
  int ImgCaffeInputFileConn::images_to_db(const std::vector<std::string> &rfolders,
					  const std::string &traindbname,
					  const std::string &testdbname,
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
	LOG(WARNING) << "image db file " << dbfullname << " already exists, bypassing creation but checking on records";
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	_db_batchsize = db->Count();
	LOG(WARNING) << "image db file " << dbfullname << " with " << _db_batchsize << " records";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "image db file " << testdbfullname << " already exists, bypassing creation but checking on records";
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    LOG(WARNING) << "image db file " << testdbfullname << " with " << _db_testbatchsize << " records";
	  }
	return 0;
      }

    // list directories in dataset train folder
    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(rfolders.at(0),false,true,subdirs))
      throw InputConnectorBadParamException("failed reading image train data directory " + rfolders.at(0));

    // list files and classes, possibly shuffle / split them
    int cl = 0;
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    std::unordered_map<std::string,int> hcorresp_r; // reverse correspondence for test set.
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
    auto uit = subdirs.begin();
    while(uit!=subdirs.end())
      {
	std::unordered_set<std::string> subdir_files;
	if (fileops::list_directory((*uit),true,false,subdir_files))
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
    else if (rfolders.size() > 1)
      {
	// list directories in dataset test folder
	std::unordered_set<std::string> test_subdirs;
	if (fileops::list_directory(rfolders.at(1),false,true,test_subdirs))
	  throw InputConnectorBadParamException("failed reading image test data directory " + rfolders.at(1));

	// list files and classes, possibly shuffle / split them
	std::unordered_map<std::string,int>::const_iterator hcit;
	auto uit = test_subdirs.begin();
	while(uit!=test_subdirs.end())
	  {
	    std::unordered_set<std::string> subdir_files;
	    if (fileops::list_directory((*uit),true,false,subdir_files))
	      throw InputConnectorBadParamException("failed reading image test data sub-directory " + (*uit));
	    std::string cls = dd_utils::split((*uit),'/').back();
	    if ((hcit=hcorresp_r.find(cls))==hcorresp_r.end())
	      {
		LOG(ERROR) << "class " << cls << " appears in testing set but not in training set, skipping";
		++uit;
		continue;
	      }
	    int cl = (*hcit).second;
	    auto fit = subdir_files.begin();
	    while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
	      {
		test_lfiles.push_back(std::pair<std::string,int>((*fit),cl));
		++fit;
	      }
	    ++uit;
	  }	
      }
    _db_batchsize = lfiles.size();
    _db_testbatchsize = test_lfiles.size();
    
    LOG(INFO) << "A total of " << lfiles.size() << " images.";
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
    Datum datum;
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    
    for (int line_id = 0; line_id < (int)lfiles.size(); ++line_id) {
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
	// Guess the encoding type from the file name
	std::string fn = lfiles[line_id].first;
	size_t p = fn.rfind('.');
	if ( p == fn.npos )
	  LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
	enc = fn.substr(p);
	std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      }
      status = ReadImageToDatum(lfiles[line_id].first,
				lfiles[line_id].second, _height, _width, !_bw,
				enc, &datum);
      if (status == false) continue;
      
      // sequential
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
			    lfiles[line_id].first.c_str());
      
      // put in db
      std::string out;
      if(!datum.SerializeToString(&out))
	LOG(ERROR) << "Failed serialization of datum for db storage";
      txn->Put(string(key_cstr, length), out);
      
      if (++count % 1000 == 0) {
	// commit db
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
  }

  int ImgCaffeInputFileConn::compute_images_mean(const std::string &dbname,
						 const std::string &meanfile,
						 const std::string &backend)
  {
    std::string dbfullname = dbname + "." + backend;
    if (fileops::file_exists(meanfile))
      {
	LOG(WARNING) << "image mean file " << meanfile << " already exists, bypassing creation";
	BlobProto sum_blob;
	ReadProtoFromBinaryFile(meanfile.c_str(),&sum_blob);
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	_mean_values = std::vector<float>(channels,0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
	  for (int i = 0; i < dim; ++i) {
	    _mean_values[c] += sum_blob.data(dim * c + i);
	  }
	  LOG(INFO) << "mean_value channel [" << c << "]:" << _mean_values[c] / dim;
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
      LOG(INFO) << "Decoding Datum";
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
	LOG(INFO) << "Processed " << count << " files.";
      }
      cursor->Next();
    }
    
    if (count % 10000 != 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    for (int i = 0; i < sum_blob.data_size(); ++i) {
      sum_blob.set_data(i, sum_blob.data(i) / count);
    }
    // Write to disk
    LOG(INFO) << "Write to " << meanfile;
    WriteProtoToBinaryFile(sum_blob, meanfile.c_str());

    // let's store the simpler mean values in case of image
    // size changes, e.g. cropping
    const int channels = sum_blob.channels();
    const int dim = sum_blob.height() * sum_blob.width();
    _mean_values = std::vector<float>(channels,0.0);
    LOG(INFO) << "Number of channels: " << channels;
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < dim; ++i) {
	_mean_values[c] += sum_blob.data(dim * c + i);
      }
      LOG(INFO) << "mean_value channel [" << c << "]:" << _mean_values[c] / dim;
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
	DecodeDatumNative(&datum);

	// deal with the mean image values, this forces to turn the datum
	// data into an array of floats (as opposed to original bytes format)
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
  
  void ImgCaffeInputFileConn::reset_dv_test()
  {
    _dt_vit = _dv_test.begin();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db = std::unique_ptr<caffe::db::DB>();
  }


  /*- DDCCsv -*/
  int DDCCsv::read_file(const std::string &fname)
  {
    if (_cifc)
      {
	_cifc->read_csv(_adconf,fname);
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
	LOG(WARNING) << "CSV db file " << dbfullname << " already exists, bypassing creation but checking on records";
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
	LOG(INFO) << "CSV db train file " << dbfullname << " with " << _db_batchsize << " records";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "CSV db file " << testdbfullname << " already exists, bypassing creation but checking on records";
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    LOG(INFO) << "CSV db test file " << testdbfullname << " with " << _db_testbatchsize << " records";
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
	LOG(INFO) << "Failed serialization of datum for db storage";
	return;
      }
    _txn->Put(std::string(key_cstr, length), out);
    _db_batchsize++;
    
    if (++count % 10000 == 0) {
      // commit db
      _txn->Commit();
      _txn.reset(_tdb->NewTransaction());
      LOG(INFO) << "Processed " << count << " records";
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
	LOG(INFO) << "Failed serialization of datum for db storage";
	return;
      }
    _ttxn->Put(std::string(key_cstr, length), out);
    _db_testbatchsize++;

    if (++count % 10000 == 0) {
      // commit db
      _ttxn->Commit();
      _ttxn.reset(_ttdb->NewTransaction());
      LOG(INFO) << "Processed " << count << " records";
    }
  }
  
  void CSVCaffeInputFileConn::write_csvline_to_db(const std::string &dbfullname,
						  const std::string &testdbfullname,
						  const APIData &ad_input,
						  const std::string &backend)
  {
    std::cerr << "CSV line to db\n";
    std::cerr << "dbfullname=" << dbfullname << std::endl;

    // Create new DB
    _tdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _tdb->Open(dbfullname.c_str(), db::NEW);
    _txn = std::unique_ptr<db::Transaction>(_tdb->NewTransaction());
    _ttdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _ttdb->Open(testdbfullname.c_str(), db::NEW);
    _ttxn = std::unique_ptr<db::Transaction>(_ttdb->NewTransaction());
    std::cerr << "db is opened\n";

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
    ddcsv.read_element(_csv_fname);

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
	LOG(WARNING) << "Txt db file " << dbfullname << " already exists, bypassing creation but checking on records";
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
	LOG(INFO) << "Txt db train file " << dbfullname << " with " << _db_batchsize << " records";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "Txt db file " << testdbfullname << " already exists, bypassing creation but checking on records";
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    LOG(INFO) << "Txt db test file " << testdbfullname << " with " << _db_testbatchsize << " records";
	  }
	// XXX: remove in-memory data, which pre-processing is useless and should be avoided
	destroy_txt_entries(_txt);
	destroy_txt_entries(_test_txt);
	
	return 0;
      }
    
    _db_batchsize = _txt.size();
    _db_testbatchsize = _test_txt.size();

    LOG(INFO) << "db_batchsize=" << _db_batchsize << " / db_testbatchsize=" << _db_testbatchsize << std::endl;
    
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
	  LOG(ERROR) << "Failed serialization of datum for db storage";
	txn->Put(string(key_cstr,length),out);

	if (++count % 1000 == 0) {
	  // commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  LOG(INFO) << "Processed " << count << " text entries";
	}
	
	++hit;
	++n;
      }

    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << count << " text entries";
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
	  LOG(ERROR) << "Failed serialization of datum for db storage";
	txn->Put(string(key_cstr,length),out);

	if (++count % 1000 == 0) {
	  // commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  LOG(INFO) << "Processed " << count << " text entries";
	}
	
	++hit;
	++n;
      }

    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << count << " text entries";
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
	LOG(WARNING) << "SVM db file " << dbfullname << " already exists, bypassing creation but checking on records";
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
	LOG(INFO) << "SVM db train file " << dbfullname << " with " << _db_batchsize << " records";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "SVM db file " << testdbfullname << " already exists, bypassing creation but checking on records";
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    _db_testbatchsize = tdb->Count();
	    LOG(INFO) << "SVM db test file " << testdbfullname << " with " << _db_testbatchsize << " records";
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
	LOG(INFO) << "Failed serialization of datum for db storage";
	return;
      }
    _txn->Put(std::string(key_cstr, length), out);
    _db_batchsize++;
    
    if (count % 10000 == 0) {
      // commit db
      _txn->Commit();
      _txn.reset(_tdb->NewTransaction());
      LOG(INFO) << "Processed " << count << " records";
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
	LOG(INFO) << "Failed serialization of datum for db storage";
	return;
      }
    _ttxn->Put(std::string(key_cstr, length), out);
    _db_testbatchsize++;

    if (count % 10000 == 0) {
      // commit db
      _ttxn->Commit();
      _ttxn.reset(_ttdb->NewTransaction());
      LOG(INFO) << "Processed " << count << " records";
    }
  }
  
  void SVMCaffeInputFileConn::write_svmline_to_db(const std::string &dbfullname,
						  const std::string &testdbfullname,
						  const APIData &ad_input,
						  const std::string &backend)
  {
    LOG(INFO) << "SVM line to db / " << "dbfullname=" << dbfullname;

    // Create new DB
    _tdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _tdb->Open(dbfullname.c_str(), db::NEW);
    _txn = std::unique_ptr<db::Transaction>(_tdb->NewTransaction());
    _ttdb = std::unique_ptr<db::DB>(db::GetDB(backend));
    _ttdb->Open(testdbfullname.c_str(), db::NEW);
    _ttxn = std::unique_ptr<db::Transaction>(_ttdb->NewTransaction());
    LOG(INFO) << "dbs " << dbfullname << " / " << testdbfullname << " opened";

    _svm_fname = _uris.at(0); // training only from file
    if (!fileops::file_exists(_svm_fname))
      throw InputConnectorBadParamException("training SVM file " + _svm_fname + " does not exist");
    if (_uris.size() > 1)
      _svm_test_fname = _uris.at(1);

    DataEl<DDSvm> ddsvm;
    ddsvm._ctype._cifc = this;
    ddsvm._ctype._adconf = ad_input;
    ddsvm.read_element(_svm_fname);

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
