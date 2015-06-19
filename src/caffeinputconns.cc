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
 * This code is adapted from Caffe's convert_imageset tool
 */

#include "caffeinputconns.h"
#include "utils/fileops.hpp"
#include "utils/utils.hpp"
#include <memory>
#include <glog/logging.h>

using namespace caffe;

namespace dd
{
  
  // convert images into db entries
  // a root folder must contain directories as classes holding image
  // files for each class. The name of the class is the name of the directory.
  int ImgCaffeInputFileConn::images_to_db(const std::string &rfolder,
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
    // however, some generic values need to be acquired, and since Caffe does not
    // provide a db size operator, we iterate and count elements.
    if (fileops::file_exists(dbfullname))
      {
	LOG(WARNING) << "image db file " << dbfullname << " already exists, bypassing creation but checking on records, may take a while...\n";
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	_db_batchsize = 0;
	while(cursor->valid())
	  {
	    ++_db_batchsize;
	    cursor->Next();
	  }
	LOG(WARNING) << "image db file " << dbfullname << " with " << _db_batchsize << " records\n";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "image db file " << testdbfullname << " already exists, bypassing creation but checking on records, may take a while...\n";
	    _test_labels.clear();
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    std::unique_ptr<db::Cursor> tcursor(tdb->NewCursor());
	    _db_testbatchsize = 0;
	    while(tcursor->valid())
	      {
		Datum datum;
		datum.ParseFromString(tcursor->value());
		_test_labels.push_back(datum.label());
		++_db_testbatchsize;
		tcursor->Next();
	      }
	    LOG(WARNING) << "image db file " << testdbfullname << " with " << _db_testbatchsize << " records\n";
	  }
	return 0;
      }

    // list directories in dataset root folder
    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(rfolder,false,true,subdirs))
      throw InputConnectorBadParamException("failed reading image data directory " + rfolder);

    // list files and classes, possibly shuffle / split them
    int cl = 0;
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
    auto uit = subdirs.begin();
    while(uit!=subdirs.end())
      {
	std::unordered_set<std::string> subdir_files;
	if (fileops::list_directory((*uit),true,false,subdir_files))
	  throw InputConnectorBadParamException("failed reading image data sub-directory " + (*uit));
	hcorresp.insert(std::pair<int,std::string>(cl,dd_utils::split((*uit),'/').back()));
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
	std::random_device rd;
	std::mt19937 g(rd());
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
    _db_batchsize = lfiles.size();
    _db_testbatchsize = test_lfiles.size();
    
    LOG(INFO) << "A total of " << lfiles.size() << " images.";
    if (lfiles.empty())
      throw InputConnectorBadParamException("no image data found in repository");
    
    // write files to dbs (i.e. train and possibly test)
    std::vector<int> labels; // XXX: useless
    write_image_to_db(dbfullname,lfiles,labels,backend,encoded,encode_type);
    if (!test_lfiles.empty())
      write_image_to_db(testdbfullname,test_lfiles,_test_labels,backend,encoded,encode_type);

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
						std::vector<int> &datum_labels,
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
	LOG(INFO) << "Failed serialization of datum for db storage";
      txn->Put(string(key_cstr, length), out);
      
      // adding label to external set (used in measures in caffelib)
      datum_labels.push_back(datum.label());

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
	LOG(WARNING) << "image mean file " << meanfile << " already exists, bypassing creation\n";
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
      /*CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;*/
      if (data.size() != 0) {
	//CHECK_EQ(data.size(), size_in_datum);
	for (int i = 0; i < size_in_datum; ++i) {
	  sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
	}
      } else {
	//CHECK_EQ(datum.float_data_size(), size_in_datum);
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
    
    const int channels = sum_blob.channels();
    const int dim = sum_blob.height() * sum_blob.width();
    std::vector<float> mean_values(channels, 0.0);
    LOG(INFO) << "Number of channels: " << channels;
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < dim; ++i) {
	mean_values[c] += sum_blob.data(dim * c + i);
      }
      LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
    }
    return 0;
  }

  std::vector<caffe::Datum> ImgCaffeInputFileConn::get_dv_test(const int &num,
							       const bool &has_mean_file)
  {
    static Blob<float> data_mean;
    static float *mean = nullptr;
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_model_repo + "/" + _test_dbfullname.c_str(),db::READ);
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
	if (i == num)
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
	_test_db_cursor->Next();
	++i;
      }
    return dv;
  }
  
  void ImgCaffeInputFileConn::reset_dv_test()
  {
    //_test_db->Close();
    _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
    _test_db = std::unique_ptr<caffe::db::DB>();
  }


  /*- CSVCaffeInputFileConn -*/
  int CSVCaffeInputFileConn::csv_to_db(const std::string &traindbname,
				       const std::string &testdbname,
				       const std::string &backend)
  {
    std::cerr << "CSV db\n";
    std::string dbfullname = traindbname + "." + backend;
    std::string testdbfullname = testdbname + "." + backend;
    
    std::cerr << "dbfullname=" << dbfullname << std::endl;
    std::cerr << "testdbfullname=" << testdbfullname << std::endl;

    // test whether the train / test dbs are already in
    // since they may be long to build, we pass on them if there already
    // in the model repository.
    // however, some generic values need to be acquired, and since Caffe does not
    // provide a db size operator, we iterate and count elements.
    if (fileops::file_exists(dbfullname))
      {
	LOG(WARNING) << "CSV db file " << dbfullname << " already exists, bypassing creation but checking on records, may take a while...\n";
	std::unique_ptr<db::DB> db(db::GetDB(backend));
	db->Open(dbfullname.c_str(), db::READ);
	std::unique_ptr<db::Cursor> cursor(db->NewCursor());
	_db_batchsize = 0;
	while(cursor->valid())
	  {
	    ++_db_batchsize;
	    cursor->Next();
	  }
	_csvdata.clear();
	LOG(WARNING) << "CSV db train file " << dbfullname << " with " << _db_batchsize << " records\n";
	if (!testdbname.empty() && fileops::file_exists(testdbfullname))
	  {
	    LOG(WARNING) << "CSV db file " << testdbfullname << " already exists, bypassing creation but checking on records, may take a while...\n";
	    _test_labels.clear();
	    std::unique_ptr<db::DB> tdb(db::GetDB(backend));
	    tdb->Open(testdbfullname.c_str(), db::READ);
	    std::unique_ptr<db::Cursor> tcursor(tdb->NewCursor());
	    _db_testbatchsize = 0;
	    while(tcursor->valid())
	      {
		Datum datum;
		datum.ParseFromString(tcursor->value());
		_test_labels.push_back(datum.label());
		++_db_testbatchsize;
		tcursor->Next();
	      }
	    LOG(WARNING) << "CSV db test file " << testdbfullname << " with " << _db_testbatchsize << " records\n";
	    _csvdata_test.clear();
	  }
	
	return 0;
      }
    
    _db_batchsize = _csvdata.size();
    _db_testbatchsize = _csvdata_test.size();
    
    // write files to dbs (i.e. train and possibly test)
    std::vector<int> labels; // XXX: useless
    write_csvline_to_db(dbfullname,_csvdata);
    _csvdata.clear();
    if (!_csvdata_test.empty())
      {
	write_csvline_to_db(testdbfullname,_csvdata_test,true);
	_csvdata_test.clear();
      }
    
    // write corresp file
    /*std::ofstream correspf(_model_repo + "/" + _correspname,std::ios::binary);
    auto hit = hcorresp.begin();
    while(hit!=hcorresp.end())
      {
	correspf << (*hit).first << " " << (*hit).second << std::endl;
	++hit;
      }
      correspf.close();*/
    
    return 0;
  }

  void CSVCaffeInputFileConn::write_csvline_to_db(const std::string &dbfullname,
						  std::vector<CSVline> &csvdata,
						  const bool &test,
						  const std::string &backend)
  {
    std::cerr << "CVS line to db\n";
    std::cerr << "dbfullname=" << dbfullname << std::endl;

    // Create new DB
    std::unique_ptr<db::DB> db(db::GetDB(backend));
    db->Open(dbfullname.c_str(), db::NEW);
    std::unique_ptr<db::Transaction> txn(db->NewTransaction());

    std::cerr << "db is opened\n";

    // Storing to db
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    int i = 0;
    auto hit = csvdata.begin();
    while(hit!=csvdata.end())
      {
	Datum d = to_datum((*hit)._v);
	
	// sequential
	int length = snprintf(key_cstr,kMaxKeyLength,"%s",std::to_string(i).c_str());
	
	// put in db
	std::string out;
	if(!d.SerializeToString(&out))
	  {
	    LOG(INFO) << "Failed serialization of datum for db storage";
	    continue;
	  }
	txn->Put(std::string(key_cstr, length), out);
	if (test)
	  _test_labels.push_back(d.label());
	
	if (++count % 1000 == 0) {
	  // commit db
	  txn->Commit();
	  txn.reset(db->NewTransaction());
	  LOG(INFO) << "Processed " << count << " records.";
	}
	
	++hit;
	++i;
      }
    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << count << " records.";
    }
    db->Close();
  }

  std::vector<caffe::Datum> CSVCaffeInputFileConn::get_dv_test_db(const int &num)
  {
    if (!_test_db_cursor)
      {
	// open db and create cursor
	if (!_test_db)
	  {
	    _test_db = std::unique_ptr<db::DB>(db::GetDB("lmdb"));
	    _test_db->Open(_model_repo + "/" + _test_dbfullname.c_str(),db::READ);
	  }
	_test_db_cursor = std::unique_ptr<db::Cursor>(_test_db->NewCursor());
      }
    std::vector<caffe::Datum> dv;
    int i =0;
    while(_test_db_cursor->valid())
      {
	// fill up a vector up to 'num' elements.
	if (i == num)
	  break;
	Datum datum;
	datum.ParseFromString(_test_db_cursor->value());
	DecodeDatumNative(&datum);
	dv.push_back(datum);
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

}
