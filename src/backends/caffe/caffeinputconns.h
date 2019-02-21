/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
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

#ifndef CAFFEINPUTCONNS_H
#define CAFFEINPUTCONNS_H

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "csvtsinputfileconn.h"
#include "txtinputfileconn.h"
#include "svminputfileconn.h"
#include "caffe/caffe.hpp"
#include "caffe/util/db.hpp"
#include "utils/fileops.hpp"

namespace dd
{
  /**
   * \brief high-level data structure shared among Caffe-compatible connectors of DeepDetect
   */
  class CaffeInputInterface
  {
  public:
    CaffeInputInterface() {}
    CaffeInputInterface(const CaffeInputInterface &cii)
      :_db(cii._db),_dv(cii._dv),_dv_test(cii._dv_test),_ids(cii._ids),_flat1dconv(cii._flat1dconv),_has_mean_file(cii._has_mean_file),_mean_values(cii._mean_values),_sparse(cii._sparse),_embed(cii._embed),_sequence_txt(cii._sequence_txt),_max_embed_id(cii._max_embed_id),_segmentation(cii._segmentation),_bbox(cii._bbox),_multi_label(cii._multi_label),_ctc(cii._ctc),_autoencoder(cii._autoencoder),_alphabet_size(cii._alphabet_size),_root_folder(cii._root_folder),_dbfullname(cii._dbfullname),_test_dbfullname(cii._test_dbfullname), _timesteps(cii._timesteps), _datadim(cii._datadim), _ntargets(cii._ntargets) {}

    ~CaffeInputInterface() {}

    /**
     * \brief when using db, this provide a batch iterator to db data,
     *        used in measuring the output of the net
     * @param num the size of the data 'batch' to get from the db
     * @param has_mean_file flag that tells whether the mean of images in the training set is removed from each image.
     * @return a vector of Caffe Datum
     * @see ImgCaffeInputFileConn
     */
    std::vector<caffe::Datum> get_dv_test(const int &num,
					  const bool &has_mean_file)
      {
	(void)has_mean_file;
	  return std::vector<caffe::Datum>(num);
      }
    
    std::vector<caffe::SparseDatum> get_dv_test_sparse(const int &num)
      {
	return std::vector<caffe::SparseDatum>(num);
      }

    void reset_dv_test() {}

    // write class weights to binary proto
    void write_class_weights(const std::string &model_repo,
			     const APIData &ad_mllib);

    bool _db = false; /**< whether to use a db. */
    std::vector<caffe::Datum> _dv; /**< main input datum vector, used for training or prediction */
    std::vector<caffe::Datum> _dv_test; /**< test input datum vector, when applicable in training mode */
    std::vector<caffe::SparseDatum> _dv_sparse;
    std::vector<caffe::SparseDatum> _dv_test_sparse;
    std::vector<std::string> _ids; /**< input ids (e.g. image ids) */
    bool _flat1dconv = false; /**< whether a 1D convolution model. */
    bool _has_mean_file = false; /**< image model mean.binaryproto. */
    std::vector<float> _mean_values; /**< mean image values across a dataset. */
    bool _sparse = false; /**< whether to use sparse representation. */
    bool _embed = false; /**< whether model is using an input embedding layer. */
    int _sequence_txt = -1; /**< sequence of txt input connector. */
    int _max_embed_id = -1; /**< in embeddings, the max index. */
    bool _segmentation = false; /**< whether it is a segmentation service. */
    bool _bbox = false; /**< whether it is an object detection service. */
    bool _multi_label = false; /**< multi label setup */
    bool _ctc = false; /**< whether it is a CTC / OCR service. */
    bool _autoencoder = false; /**< whether an autoencoder. */
    int _alphabet_size = 0; /**< for sequence to sequence models. */
    std::string _root_folder; /**< root folder for image list layer. */
    std::unordered_map<std::string,std::pair<int,int>> _imgs_size; /**< image sizes, used in detection. */
    std::string _dbfullname = "train.lmdb";
    std::string _test_dbfullname = "test.lmdb";
    int _timesteps = -1;  //default length for csv timeseries
    int _datadim = -1; //default size of vector data for timeseries
    int _ntargets = -1; // number of outputs for timeseries
  };

  /**
   * \brief Caffe image connector, supports both files and building of database for training
   */
  class ImgCaffeInputFileConn : public ImgInputFileConn, public CaffeInputInterface
  {
  public:
    ImgCaffeInputFileConn()
      :ImgInputFileConn() {
      reset_dv_test();
    }
    ImgCaffeInputFileConn(const ImgCaffeInputFileConn &i)
      :ImgInputFileConn(i),CaffeInputInterface(i) {/* _db = true;*/ }
    ~ImgCaffeInputFileConn() {}

    // size of each element in Caffe jargon
    int channels() const
    {
      if (_bw) return 1;
      else return 3; // RGB
    }
    
    int height() const
    {
      return _height;
    }
    
    int width() const
    {
      return _width;

    }

    int batch_size() const
    {
      if (_db_batchsize > 0)
	return _db_batchsize;
      else if (!_dv.empty())
	return _dv.size();
      else return ImgInputFileConn::batch_size();
    }
    
    int test_batch_size() const
    {
      if (_db_testbatchsize > 0)
	return _db_testbatchsize;
      else if (!_dv_test.empty())
	return _dv_test.size();
      else return ImgInputFileConn::test_batch_size();
    }

    void init(const APIData &ad)
    {
      ImgInputFileConn::init(ad);
      if (ad.has("db"))
	_db = ad.get("db").get<bool>();
      if (ad.has("multi_label"))
	_multi_label = ad.get("multi_label").get<bool>();
      if (ad.has("root_folder"))
	_root_folder = ad.get("root_folder").get<std::string>();
      if (ad.has("segmentation"))
	_segmentation = ad.get("segmentation").get<bool>();
      if (ad.has("bbox"))
	_bbox = ad.get("bbox").get<bool>();
      if (ad.has("ctc"))
	_ctc = ad.get("ctc").get<bool>();
    }

    void transform(const APIData &ad)
    {
      // in prediction mode, convert the images to Datum, a Caffe data structure
      if (!_train)
	{
	  // if no img height x width, we assume 224x224 (works if user is lucky, i.e. the best we can do)
	  if (_width == -1)
	    _width = 224;
	  if (_height == -1)
	    _height = 224;
	  
	  if (ad.has("has_mean_file"))
	    _has_mean_file = ad.get("has_mean_file").get<bool>();
	  APIData ad_input = ad.getobj("parameters").getobj("input");
	  if (ad_input.has("segmentation"))
	    _segmentation = ad_input.get("segmentation").get<bool>();
	  if (ad_input.has("bbox"))
	    _bbox = ad_input.get("bbox").get<bool>();
	  if (ad_input.has("multi_label"))
	    _multi_label = ad_input.get("multi_label").get<bool>();
	  if (ad.has("root_folder"))
	    _root_folder = ad.get("root_folder").get<std::string>();
	  try
	    {
	      ImgInputFileConn::transform(ad);
	    }
	  catch (InputConnectorBadParamException &e)
	    {
	      throw;
	    }
	  float *mean = nullptr;
	  std::string meanfullname = _model_repo + "/" + _meanfname;
	  if (_data_mean.count() == 0 && _has_mean_file)
	    {
	      caffe::BlobProto blob_proto;
	      caffe::ReadProtoFromBinaryFile(meanfullname.c_str(),&blob_proto);
	      _data_mean.FromProto(blob_proto);
	      mean = _data_mean.mutable_cpu_data();
	    }
	  if (!_db_fname.empty())
	    {
	      _test_dbfullname = _db_fname;
	      _db = true;
	      return; // done
	    }
	  else _db = false;
	  for (int i=0;i<(int)this->_images.size();i++)
	    {      
	      caffe::Datum datum;
	      caffe::CVMatToDatum(this->_images.at(i),&datum);
	      if (!_test_labels.empty())
		datum.set_label(_test_labels.at(i));
	      if (_data_mean.count() != 0)
		{
		  int height = datum.height();
		  int width = datum.width();
		  for (int c=0;c<datum.channels();++c)
		    for (int h=0;h<height;++h)
		      for (int w=0;w<width;++w)
			{
			  int data_index = (c*height+h)*width+w;
			  float datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
			  datum.add_float_data(datum_element - mean[data_index]);
			}
		  datum.clear_data();
		}
	      else if (_has_mean_scalar)
		{
		  int height = datum.height();
		  int width = datum.width();
		  for (int c=0;c<datum.channels();++c)
		    for (int h=0;h<height;++h)
		      for (int w=0;w<width;++w)
			{
			  int data_index = (c*height+h)*width+w;
			  float datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
			  datum.add_float_data(datum_element - _mean[c]);
			}
		  datum.clear_data();
		}
	      _dv_test.push_back(datum);
	      _ids.push_back(this->_uris.at(i));
	      _imgs_size.insert(std::pair<std::string,std::pair<int,int>>(this->_uris.at(i),this->_images_size.at(i)));
	    }
	  this->_images.clear();
	  this->_images_size.clear();
	}
      else
	{
	  _shuffle = true;
	  int db_height = 0;
	  int db_width = 0;
	  APIData ad_mllib;
	  if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	    {
	      APIData ad_param = ad.getobj("parameters");
	      if (ad_param.has("input"))
		{
		  APIData ad_input = ad_param.getobj("input");
		  fillup_parameters(ad_param.getobj("input"));
		  if (ad_input.has("db"))
		    _db = ad_input.get("db").get<bool>();
		  if (ad_input.has("segmentation"))
		    _segmentation = ad_input.get("segmentation").get<bool>();
		  if (ad_input.has("bbox"))
		    _bbox = ad_input.get("bbox").get<bool>();
		  if (ad_input.has("multi_label"))
		    _multi_label = ad_input.get("multi_label").get<bool>();
		  if (ad_input.has("root_folder"))
		    _root_folder = ad_input.get("root_folder").get<std::string>();
		  if (ad_input.has("align"))
		    _align = ad_input.get("align").get<bool>();
		  if (ad_input.has("db_height"))
		    db_height = ad_input.get("db_height").get<int>();
		  if (ad_input.has("db_width"))
		    db_width = ad_input.get("db_width").get<int>();
		  if (ad.has("autoencoder"))
		    _autoencoder = ad.get("autoencoder").get<bool>();
		}
	      ad_mllib = ad_param.getobj("mllib");
	    }

	  if (_segmentation)
	    {
	      try
		{
		  get_data(ad);
		}
	      catch(InputConnectorBadParamException &ex)
		{
		  throw ex;
		}
	      if (!fileops::file_exists(_uris.at(0)))
		throw InputConnectorBadParamException("segmentation input train file " + _uris.at(0) + " not found");
	      if (_uris.size() > 1)
		{
		  if (!fileops::file_exists(_uris.at(1)))
		    throw InputConnectorBadParamException("segmentation input test file " + _uris.at(1) + " not found");
		}

	      // class weights if any
	      write_class_weights(_model_repo,ad_mllib);
	      
	      //TODO: if test split (+ optional shuffle)
	      APIData sourcead;
	      sourcead.add("source_train",_uris.at(0));
	      if (_uris.size() > 1)
            sourcead.add("source_test",_uris.at(1));
	      const_cast<APIData&>(ad).add("source",sourcead);
	    }
	  else if (_bbox)
	    {
	      try
		{
		  get_data(ad);
		}
	      catch(InputConnectorBadParamException &ex)
		{
		  throw ex;
		}
	      if (!fileops::file_exists(_uris.at(0)))
		throw InputConnectorBadParamException("object detection input train file " + _uris.at(0) + " not found");
	      if (_uris.size() > 1)
		{
		  if (!fileops::file_exists(_uris.at(1)))
		    throw InputConnectorBadParamException("object detection input test file " + _uris.at(1) + " not found");
		}

	      // - create lmdbs
	      _dbfullname = _model_repo + "/" + _dbfullname;
	      _test_dbfullname = _model_repo + "/" + _test_dbfullname;
	      objects_to_db(_uris,db_height,db_width,_dbfullname,_test_dbfullname);
	      
	      // data object with db files location
	      APIData dbad;
	      dbad.add("train_db",_dbfullname);
	      if (_test_split > 0.0 || _uris.size() > 1)
		dbad.add("test_db",_test_dbfullname);
	      const_cast<APIData&>(ad).add("db",dbad); 
	    }
	  else if (_ctc)
	    {
	      _dbfullname = _model_repo + "/train";
	      _test_dbfullname = _model_repo + "/test.h5";
	      try
		{
		  get_data(ad);
		}
	      catch(InputConnectorBadParamException &ex) // in case the db is in the net config
		{
		  throw ex;
		}

	      // read images list and create dbs
#ifdef USE_HDF5
	      images_to_hdf5(_uris,_dbfullname,_test_dbfullname);
#endif // USE_HDF5
	      // enrich data object with db files location
	      APIData dbad;
	      dbad.add("train_db",_model_repo + "/training.txt");
	      if (_uris.size() > 1 || _test_split > 0.0)
		dbad.add("test_db",_model_repo + "/testing.txt");
	      const_cast<APIData&>(ad).add("db",dbad);
	    }
	  else // more complicated, since images can be heavy, a db is built so that it is less costly to iterate than the filesystem, unless image data layer is used (e.g. multi-class image training)
	    {
	      _dbfullname = _model_repo + "/" + _dbfullname;
	      _test_dbfullname = _model_repo + "/" + _test_dbfullname;
	      try
		{
		  get_data(ad);
		}
	      catch(InputConnectorBadParamException &ex) // in case the db is in the net config
		{
		  // API defines no data as a user error (bad param).
		  // However, Caffe does allow to specify the input database into the net's definition,
		  // which makes it difficult to enforce the API here.
		  // So for now, this check is kept disabled.
		  /*if (!fileops::file_exists(_model_repo + "/" + _dbname))
		    throw ex;*/
		  return;
		}
	      if (!this->_db) {
		// create test db for image data layer (no db of images)
		create_test_db_for_imagedatalayer(_uris.at(1),_model_repo + "/" +_test_dbname);
		return;
	      }
	      // create db
	      // Check if the indicated uri is a folder
	      bool dir_images = true;
	      fileops::file_exists(_uris.at(0), dir_images);
	      if (!this->_unchanged_data)
	        images_to_db(_uris,_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname, dir_images);
	      else 
	        images_to_db(_uris,_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname, dir_images,
					"lmdb",false,"");

	      // compute mean of images, not forcely used, depends on net, see has_mean_file
	      if (!this->_unchanged_data)
		compute_images_mean(_model_repo + "/" + _dbname,
				    _model_repo + "/" + _meanfname);
	      
	      // class weights if any
	      write_class_weights(_model_repo,ad_mllib);
	      
	      // enrich data object with db files location
	      APIData dbad;
	      dbad.add("train_db",_dbfullname);
	      if (_test_split > 0.0)
		dbad.add("test_db",_test_dbfullname);
	      dbad.add("meanfile",_model_repo + "/" + _meanfname);
	      const_cast<APIData&>(ad).add("db",dbad);
	    }
	}
    }

    std::vector<caffe::Datum> get_dv_test(const int &num,
					  const bool &has_mean_file)
      {
        if (_segmentation && _train)
	  {
	    return get_dv_test_segmentation(num,has_mean_file);
	  }
	else if (!_train && _db_fname.empty())
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
	else return get_dv_test_db(num,has_mean_file);
      }
    
    std::vector<caffe::Datum> get_dv_test_db(const int &num,
					     const bool &has_mean_file);

    std::vector<caffe::Datum> get_dv_test_segmentation(const int &num,
						       const bool &has_mean_file);
    
    void reset_dv_test();
    
  private:

    void create_test_db_for_imagedatalayer(const std::string &test_lst,
                                           const std::string &testdbname,
                                           const std::string &backend="lmdb", // lmdb, leveldb
                                           const bool &encoded=true, // save the encoded image in datum
                                           const std::string &encode_type=""); // 'png', 'jpg', ...

    int images_to_db(const std::vector<std::string> &rpaths,
		     const std::string &traindbname,
		     const std::string &testdbname,
		     const bool &folders=true,						 
		     const std::string &backend="lmdb", // lmdb, leveldb
		     const bool &encoded=true, // save the encoded image in datum
		     const std::string &encode_type=""); // 'png', 'jpg', ...

    void write_image_to_db(const std::string &dbfullname,
			   const std::vector<std::pair<std::string,int>> &lfiles,
			   const std::string &backend,
			   const bool &encoded,
			   const std::string &encode_type);

    void write_image_to_db_multilabel(const std::string &dbfullname,
				      const std::vector<std::pair<std::string,std::vector<float>>> &lfiles,
				      const std::string &backend,
				      const bool &encoded,
				      const std::string &encode_type);
		#ifdef USE_HDF5
    void images_to_hdf5(const std::vector<std::string> &img_lists,
			const std::string &traindbname,
			const std::string &testdbname);

    void write_images_to_hdf5(const std::string &inputfilename,
			      const std::string &dbfullbame,
			      const std::string &dblistfilename,
			      std::unordered_map<uint32_t,int> &alphabet,
			      int &max_ocr_length,
			      const bool &train_db);
		#endif // USE_HDF5

    int objects_to_db(const std::vector<std::string> &rfolders,
		      const int &db_height,
		      const int &db_width,
		      const std::string &traindbname,
		      const std::string &testdbname,
		      const bool &encoded=true,
		      const std::string &encode_type="",
		      const std::string &backend="lmdb");

    void write_objects_to_db(const std::string &dbfullname,
			     const int &db_height,
			     const int &db_width,
			     const std::vector<std::pair<std::string,std::string>> &lines,
			     const bool &encoded,
			     const std::string &encode_type,
			     const std::string &backend,
			     const bool &train);
    
    int compute_images_mean(const std::string &dbname,
			    const std::string &meanfile,
			    const std::string &backend="lmdb");

    std::string guess_encoding(const std::string &file);
    
  public:
    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
    std::string _meanfname = "mean.binaryproto";
    std::string _correspname = "corresp.txt";
    caffe::Blob<float> _data_mean; // mean binary image if available.
    std::vector<caffe::Datum>::const_iterator _dt_vit;
    std::vector<std::pair<std::string,std::string>> _segmentation_data_lines;
    int _dt_seg = 0;
    bool _align = false;
  };

  /**
   * \brief Caffe CSV connector
   * \note use 'label_offset' in API to make sure that labels start at 0
   */
  class CSVCaffeInputFileConn;
  class DDCCsv
  {
  public:
    DDCCsv() {}
    ~DDCCsv() {}

    int read_file(const std::string &fname);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir)
    {
      throw InputConnectorBadParamException("uri " + dir + " is a directory, requires a CSV file");
    }
    
    CSVCaffeInputFileConn *_cifc = nullptr;
    APIData _adconf;
    std::shared_ptr<spdlog::logger> _logger;
  };
  
  class CSVCaffeInputFileConn : public CSVInputFileConn, public CaffeInputInterface
  {
  public:
    CSVCaffeInputFileConn()
      :CSVInputFileConn() 
      {
	reset_dv_test();
      }
    CSVCaffeInputFileConn(const CSVCaffeInputFileConn &i)
      :CSVInputFileConn(i),CaffeInputInterface(i) {}
    ~CSVCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }

    // size of each element in Caffe jargon
    int channels() const
    {
      if (_channels > 0)
	return _channels;
      return feature_size();
    }
    
    int height() const
    {
      return 1;
    }
    
    int width() const
    {
      return 1;
    }

    int batch_size() const
    {
      if (_db_batchsize > 0)
	return _db_batchsize;
      else return _dv.size();
    }

    int test_batch_size() const
    {
      if (_db_testbatchsize > 0)
	return _db_testbatchsize;
      else return _dv_test.size();
    }

    virtual void add_train_csvline(const std::string &id,
				   std::vector<double> &vals);

    virtual void add_test_csvline(const std::string &id,
				  std::vector<double> &vals);
    
    void transform(const APIData &ad)
    {
      APIData ad_param = ad.getobj("parameters");
      APIData ad_input = ad_param.getobj("input");
      APIData ad_mllib = ad_param.getobj("mllib");
      
      if (_train && ad_input.has("db") && ad_input.get("db").get<bool>())
	{
	  _dbfullname = _model_repo + "/" + _dbfullname;
	  _test_dbfullname = _model_repo + "/" + _test_dbfullname;
	  fillup_parameters(ad_input);
	  get_data(ad);
	  _db = true;
	  csv_to_db(_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname,
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
	      CSVInputFileConn::transform(ad);
	    }
	  catch (std::exception &e)
	    {
	      throw;
	    }
	  
	  // transform to datum by filling up float_data
	  if (_train)
	    {
	      auto hit = _csvdata.begin();
	      while(hit!=_csvdata.end())
		{
		  if (_label.size() == 1)
		    _dv.push_back(to_datum((*hit)._v));
		  else // multi labels or autoencoder
		    {
		      caffe::Datum dat = to_datum((*hit)._v,true);
		      for (size_t i=0;i<_label_pos.size();i++) // concat labels and slice them out in the network itself
			{
			  dat.add_float_data(static_cast<float>((*hit)._v.at(_label_pos[i])));
			}
		      dat.set_channels(dat.channels()+_label.size());
		      _dv.push_back(dat);
		    }
		  _ids.push_back((*hit)._str);
		  ++hit;
		}
	    }
	  if (!_train)
	    {
	      if (!_db_fname.empty())
		{
		  _test_dbfullname = _db_fname;
		  _db = true;
		  return; // done
		}
	      _csvdata_test = std::move(_csvdata);
	    }
	  else _csvdata.clear();
	  auto hit = _csvdata_test.begin();
	  while(hit!=_csvdata_test.end())
	    {
	      // no ids taken on the test set
	      if (_label.size() == 1)
		_dv_test.push_back(to_datum((*hit)._v));
	      else
		{
		  caffe::Datum dat = to_datum((*hit)._v,true);
		  for (size_t i=0;i<_label_pos.size();i++)
		    {
		      dat.add_float_data(static_cast<float>((*hit)._v.at(_label_pos[i])));
		    }
		  dat.set_channels(dat.channels()+_label.size());
		  _dv_test.push_back(dat);
		}
	      if (!_train)
		_ids.push_back((*hit)._str);
	      ++hit;
	    }
	  _csvdata_test.clear();
	}
      _csvdata_test.clear();
    }

    std::vector<caffe::Datum> get_dv_test(const int &num,
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
    
    std::vector<caffe::Datum> get_dv_test_db(const int &num);

    void reset_dv_test();

    /**
     * \brief turns a vector of values into a Caffe Datum structure
     * @param vector of values
     * @return datum
     */
    caffe::Datum to_datum(const std::vector<double> &vf,
			  const bool &multi_label=false)
    {
      caffe::Datum datum;
      int datum_channels = vf.size();
      if (!_label.empty())
	datum_channels -= _label.size();
      if (!_id.empty())
	datum_channels--;
      datum.set_channels(datum_channels);
      datum.set_height(1);
      datum.set_width(1);
      auto lit = _columns.begin();
      for (int i=0;i<(int)vf.size();i++)
	{
	  if (!multi_label && !this->_label.empty() && i == _label_pos[0])
	    {
	      datum.set_label(static_cast<float>(vf.at(i)+this->_label_offset[0]));
	    }
	  else if (i == _id_pos)
	    {
	      ++lit;
	      continue;
	    }
	  else if (std::find(_label_pos.begin(),_label_pos.end(),i)==_label_pos.end()) // XXX: could do a faster lookup
	  {
	      datum.add_float_data(static_cast<float>(vf.at(i)));
	    }
	  ++lit;
	}
      return datum;
    }

  private:
    int csv_to_db(const std::string &traindbname,
		  const std::string &testdbname,
		  const APIData &ad_input,
		  const std::string &backend="lmdb"); // lmdb, leveldb

    void write_csvline_to_db(const std::string &dbfullname,
			     const std::string &testdbfullname,
			     const APIData &ad_input,
			     const std::string &backend="lmdb");
    
  public:
    std::vector<caffe::Datum>::const_iterator _dt_vit;
    
    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
    std::string _correspname = "corresp.txt";

  private:
    std::unique_ptr<caffe::db::Transaction> _txn;
    std::unique_ptr<caffe::db::DB> _tdb;
    std::unique_ptr<caffe::db::Transaction> _ttxn;
    std::unique_ptr<caffe::db::DB> _ttdb;
    int _channels = 0;
  };

  /**
   * \brief caffe csv timeseries connector
   */

  class CSVTSCaffeInputFileConn;
  class DDCCsvTS
  {
  public:
  DDCCsvTS() {}
    ~DDCCsvTS() {}
    int read_file(const std::string &fname, bool is_test_data=false);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir, bool is_test_data=false);

    DDCsvTS _ddcsvts;
    CSVTSCaffeInputFileConn *_cifc = nullptr;
    APIData _adconf;
    std::shared_ptr<spdlog::logger> _logger;
  };


  class CSVTSCaffeInputFileConn : public CSVTSInputFileConn, public CaffeInputInterface
  {
  public:
  CSVTSCaffeInputFileConn()
    :CSVTSInputFileConn(), _dv_index(-1), _dv_test_index(-1), _continuation(false), _offset(100)
      {
        reset_dv_test();
      }
  CSVTSCaffeInputFileConn(const CSVTSCaffeInputFileConn &i)
    :CSVTSInputFileConn(i), CaffeInputInterface(i), _dv_index(i._dv_index), _dv_test_index(i._dv_test_index), _continuation(i._continuation), _offset(i._offset)
      {
        this->_datadim = i._datadim;
      }
    ~CSVTSCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      CSVTSInputFileConn::fillup_parameters(ad_input);
      _ntargets = _label.size();
      _offset = _timesteps;
      if (ad_input.has("timesteps"))
        {
          _timesteps= ad_input.get("timesteps").get<int>();
          _offset = _timesteps;
        }
      if (ad_input.has("continuation"))
        _continuation= ad_input.get("continuation").get<bool>();
      if (ad_input.has("offset"))
        _offset= ad_input.get("offset").get<int>();
    }


    // size of each element in Caffe jargon
    int channels() const
    {
      return _timesteps;
    }

    int height() const
    {
      return _datadim;
    }

    int width() const
    {
      return 1;
    }

    int batch_size() const
    {
      if (_db_batchsize > 0)
	return _db_batchsize;
      else if (_dv.size() != 0)
	return _dv.size();
      else
        return 1;
    }

    int test_batch_size() const
    {
      if (_db_testbatchsize > 0)
	return _db_testbatchsize;
      else if (_dv_test.size() != 0)
	return _dv_test.size();
      else
        return 1;
    }

    void push_csv_to_csvts(bool is_test_data=false);
    void set_datadim(bool is_test_data = false);
    void transform(const APIData &ad); // calls CSVTSInputfileconn::transform and db stuff
    void reset_dv_test();
    std::vector<caffe::Datum> get_dv_test(const int &num,
                                          const bool &has_mean_file);
    std::vector<caffe::Datum> get_dv_test_db(const int &num);

    int csvts_to_db(const std::string &traindbname,
                    const std::string &testdbname,
                    const APIData &ad_input,
                    const std::string &backend="lmdb"); // lmdb, leveldb
    void csvts_to_dv(bool is_test_data=false, bool clear_dv_first = false, bool clear_csvts_after=false, bool split_seqs=true, bool first_is_cont = false);
    void dv_to_db(bool is_test_data=false);

    void write_csvts_to_db(const std::string &dbfullname,
                           const std::string &testdbfullname,
                           const APIData &ad_input,
                           const std::string &backend);

    std::vector<caffe::Datum>::const_iterator _dt_vit;

    int _dv_index;
    int _dv_test_index;

    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
    std::string _correspname = "corresp.txt";
    bool _continuation;
    int _offset;

  private:
    std::unique_ptr<caffe::db::Transaction> _txn;
    std::unique_ptr<caffe::db::DB> _tdb;
    std::unique_ptr<caffe::db::Transaction> _ttxn;
    std::unique_ptr<caffe::db::DB> _ttdb;
    int _channels = 0;


  };

  /**
   * \brief Caffe text connector
   */
  class TxtCaffeInputFileConn : public TxtInputFileConn, public CaffeInputInterface
  {
  public:
    TxtCaffeInputFileConn()
      :TxtInputFileConn()
      {
	reset_dv_test();
      }
    TxtCaffeInputFileConn(const TxtCaffeInputFileConn &i)
      :TxtInputFileConn(i),CaffeInputInterface(i) {}
    ~TxtCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      TxtInputFileConn::init(ad);
      if (_characters)
	_flat1dconv = true;
      if (ad.has("sparse") && ad.get("sparse").get<bool>())
	_sparse = true;
      if (ad.has("embedding") && ad.get("embedding").get<bool>())
	_embed = true;
      _sequence_txt = _sequence;
      _max_embed_id = _alphabet.size() + 1; // +1 as offset to null index
    }

    int channels() const
    {
      if (_characters)
	return 1;
      if (_embed)
	{
	  if (!_characters)
	    return _sequence;
	  else return 1;
	}
      if (_channels > 0)
	return _channels;
      return feature_size();
    }

    int height() const
    {
      if (_characters)
	return _sequence;
      else return 1;
    }

    int width() const
    {
      if (_characters && !_embed)
	return _alphabet.size();
      return 1;
    }

    int batch_size() const
    {
      if (_db_batchsize > 0)
	return _db_batchsize;
      else if (!_sparse)
	return _dv.size();
      else return _dv_sparse.size();
    }

    int test_batch_size() const
    {
      if (_db_testbatchsize > 0)
	return _db_testbatchsize;
      else if (!_sparse)
	return _dv_test.size();
      else return _dv_test_sparse.size();
    }
    
    int txt_to_db(const std::string &traindbname,
		  const std::string &testdbname,
		  const std::string &backend="lmdb");

    void write_txt_to_db(const std::string &dbname,
			 std::vector<TxtEntry<double>*> &txt,
			 const std::string &backend="lmdb");
    
    void write_sparse_txt_to_db(const std::string &dbname,
				std::vector<TxtEntry<double>*> &txt,
				const std::string &backend="lmdb");
    
    void transform(const APIData &ad)
    {
      APIData ad_param = ad.getobj("parameters");
      APIData ad_input = ad_param.getobj("input");
      APIData ad_mllib = ad_param.getobj("mllib");
      if (ad_input.has("db") && ad_input.get("db").get<bool>())
	_db = true;
      if (ad_input.has("embedding") && ad_input.get("embedding").get<bool>())
	{
	  _embed = true;
	}
      
      // transform to one-hot vector datum
      if (_train && _db)
	{
	  _dbfullname = _model_repo + "/" + _dbfullname;
	  _test_dbfullname = _model_repo + "/" + _test_dbfullname;
	  //std::string dbfullname = _model_repo + "/" + _dbname + ".lmdb";
	  if (!fileops::file_exists(_dbfullname)) // if no existing db, preprocess from txt files
	    TxtInputFileConn::transform(ad);
	  txt_to_db(_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname);
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
	  TxtInputFileConn::transform(ad);
	  
	  if (_train)
	    {
	      auto hit = _txt.begin();
	      while(hit!=_txt.end())
		{
		  if (!_sparse)
		    {
		      if (_characters)
			_dv.push_back(std::move(to_datum<TxtCharEntry>(static_cast<TxtCharEntry*>((*hit)))));
		      else _dv.push_back(std::move(to_datum<TxtBowEntry>(static_cast<TxtBowEntry*>((*hit)))));
		    }
		  else
		    {
		      if (_characters)
			{
			  //TODO
			}
		      else _dv_sparse.push_back(std::move(to_sparse_datum(static_cast<TxtBowEntry*>((*hit)))));
		    }
		  _ids.push_back((*hit)->_uri);
		  ++hit;
		}
	    }
	  if (!_train)
	    {
	      if (!_db_fname.empty())
		{
		  _test_dbfullname = _db_fname;
		  _db = true;
		  return; // done
		}
	    _test_txt = std::move(_txt);
	    }

	  int n = 0;
	  auto hit = _test_txt.begin();
	  while(hit!=_test_txt.end())
	    {
	      if (!_sparse)
		{
		  if (_characters)
		    _dv_test.push_back(std::move(to_datum<TxtCharEntry>(static_cast<TxtCharEntry*>((*hit)))));
		  else _dv_test.push_back(std::move(to_datum<TxtBowEntry>(static_cast<TxtBowEntry*>((*hit)))));
		}
	      else
		{
		  if (_characters)
		    {
		      //TODO
		    }
		  else _dv_test_sparse.push_back(std::move(to_sparse_datum(static_cast<TxtBowEntry*>((*hit)))));
		}
	      if (!_train)
		_ids.push_back(std::to_string(n));
	      ++hit;
	      ++n;
	    }
	}
    }

    std::vector<caffe::Datum> get_dv_test_db(const int &num);
    std::vector<caffe::SparseDatum> get_dv_test_sparse_db(const int &num);
    
    std::vector<caffe::Datum> get_dv_test(const int &num,
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

    std::vector<caffe::SparseDatum> get_dv_test_sparse(const int &num)
      {
	if (!_db)
	  {
	    int i = 0;
	    std::vector<caffe::SparseDatum> dv;
	    while(_dt_vit_sparse!=_dv_test_sparse.end()
		  && i < num)
	      {
		dv.push_back((*_dt_vit_sparse));
		++i;
		++_dt_vit_sparse;
	      }
	    return dv;
	  }
      	else return get_dv_test_sparse_db(num);
      }

    void reset_dv_test()
    {
      if (!_sparse)
	_dt_vit = _dv_test.begin();
      else _dt_vit_sparse = _dv_test_sparse.begin();
      _test_db_cursor = std::unique_ptr<caffe::db::Cursor>();
      _test_db = std::unique_ptr<caffe::db::DB>();
    }
    
    template<class TEntry> caffe::Datum to_datum(TEntry *tbe)
      {
	caffe::Datum datum;
	int datum_channels;
	if (_characters)
	  datum_channels = 1;
	else if (_embed && !_characters)
	  datum_channels = _sequence;
	else datum_channels = _vocab.size(); // XXX: may be very large
	datum.set_channels(datum_channels);
       	datum.set_height(1);
	datum.set_width(1);
	datum.set_label(tbe->_target);
	if (!_characters)
	  {
	    std::unordered_map<std::string,Word>::const_iterator wit;
	    if (!_embed)
	      {
		for (int i=0;i<datum_channels;i++) // XXX: expected to be slow
		  datum.add_float_data(0.0);
		tbe->reset();
		while(tbe->has_elt())
		  {
		    std::string key;
		    double val;
		    tbe->get_next_elt(key,val);
		    if ((wit = _vocab.find(key))!=_vocab.end())
		      datum.set_float_data(_vocab[key]._pos,static_cast<float>(val));
		  }
	      }
	    else
	      {
		tbe->reset();
		int i = 0;
		while(tbe->has_elt())
		  {
		    std::string key;
		    double val;
		    tbe->get_next_elt(key,val);
		    if ((wit = _vocab.find(key))!=_vocab.end())
		      datum.add_float_data(static_cast<float>(_vocab[key]._pos));
		    ++i;
		    if (i == _sequence) // tmp limit on sequence length
		      break;
		  }
		while (datum.float_data_size() < _sequence)
		  datum.add_float_data(0.0);
	      }
	  }
	else // character-level features
	  {
	    tbe->reset();
	    std::vector<int> vals;
	    std::unordered_map<uint32_t,int>::const_iterator whit;
	    while(tbe->has_elt())
	      {
		std::string key;
		double val = -1.0;
		tbe->get_next_elt(key,val);
		uint32_t c = std::strtoul(key.c_str(),0,10);
		if ((whit=_alphabet.find(c))!=_alphabet.end())
		  vals.push_back((*whit).second);
		else vals.push_back(-1);
	      }
	    /*if (vals.size() > _sequence)
	      std::cerr << "more characters than sequence / " << vals.size() << " / sequence=" << _sequence << std::endl;*/
	    if (!_embed)
	      {
		for (int c=0;c<_sequence;c++)
		  {
		    std::vector<float> v(_alphabet.size(),0.0);
		    if (c<(int)vals.size() && vals[c] != -1)
		      v[vals[c]] = 1.0;
		    for (float f: v)
		      datum.add_float_data(f);
		  }
		datum.set_height(_sequence);
		datum.set_width(_alphabet.size());
	      }
	    else
	      {
		for (int c=0;c<_sequence;c++)
		  {
		    double val = 0.0;
		    if (c<(int)vals.size() && vals[c] != -1)
		      val = static_cast<float>(vals[c]+1.0); // +1 as offset to null index
		    datum.add_float_data(val);
		  }
		datum.set_height(_sequence);
		datum.set_width(1);
	      }
	  }
	return datum;
      }

    caffe::SparseDatum to_sparse_datum(TxtBowEntry *tbe)
      {
	caffe::SparseDatum datum;
	datum.set_label(tbe->_target);
	std::unordered_map<std::string,Word>::const_iterator wit;
	tbe->reset();
	int nwords = 0;
	while(tbe->has_elt())
	  {
	    std::string key;
	    double val;
	    tbe->get_next_elt(key,val);
	    if ((wit = _vocab.find(key))!=_vocab.end())
	      {
		int word_pos = _vocab[key]._pos;
		datum.add_data(static_cast<float>(val));
		datum.add_indices(word_pos);
		++nwords;
	      }
	  }
	datum.set_nnz(nwords);
	datum.set_size(_vocab.size());
	return datum;
      }

    std::vector<caffe::Datum>::const_iterator _dt_vit;
    std::vector<caffe::SparseDatum>::const_iterator _dt_vit_sparse;

  public:
    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
    int _channels = 0;
  };

  /**
   * \brief Caffe SVM connector
   */
  class SVMCaffeInputFileConn : public SVMInputFileConn, public CaffeInputInterface
  {
  public:
    SVMCaffeInputFileConn()
      :SVMInputFileConn()
      {
	_sparse = true;
	reset_dv_test();
      }
    SVMCaffeInputFileConn(const SVMCaffeInputFileConn &i)
      :SVMInputFileConn(i),CaffeInputInterface(i) {}
    ~SVMCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      SVMInputFileConn::init(ad);
    }

    int channels() const
    {
      if (_channels > 0)
	return _channels;
      else return feature_size();
    }

    int height() const
    {
      return 1;
    }

    int width() const
    {
      return 1;
    }

    int batch_size() const
    {
      if (_db_batchsize > 0)
	return _db_batchsize;
      else return _dv_sparse.size();
    }

    int test_batch_size() const
    {
      if (_db_testbatchsize > 0)
	return _db_testbatchsize;
      else return _dv_test_sparse.size();
    }

    virtual void add_train_svmline(const int &label,
				   const std::unordered_map<int,double> &vals,
				   const int &count);
    virtual void add_test_svmline(const int &label,
				  const std::unordered_map<int,double> &vals,
				  const int &count);

    void transform(const APIData &ad)
    {
      APIData ad_param = ad.getobj("parameters");
      APIData ad_input = ad_param.getobj("input");
      APIData ad_mllib = ad_param.getobj("mllib");
      
      if (_train && ad_input.has("db") && ad_input.get("db").get<bool>())
	{
	  _dbfullname = _model_repo + "/" + _dbfullname;
	  _test_dbfullname = _model_repo + "/" + _test_dbfullname;
	  fillup_parameters(ad_input);
	  get_data(ad);
	  _db = true;
	  svm_to_db(_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname,ad_input);
	  write_class_weights(_model_repo,ad_mllib);
	  
	  // enrich data object with db files location
	  APIData dbad;
	  dbad.add("train_db",_dbfullname);
	  if (_test_split > 0.0)
	    dbad.add("test_db",_test_dbfullname);
	  const_cast<APIData&>(ad).add("db",dbad);
	  serialize_vocab();
	}
      else
	{
	  try
	    {
	      SVMInputFileConn::transform(ad);
	    }
	  catch(std::exception &e)
	    {
	      throw;
	    }
	
	  if (_train)
	    {
	      write_class_weights(_model_repo,ad_mllib);
	      int n = 0;
	      auto hit = _svmdata.begin();
	      while(hit!=_svmdata.end())
		{
		  _dv_sparse.push_back(to_sparse_datum((*hit)));
		  _ids.push_back(std::to_string(n));
		  ++n;
		  ++hit;
		}
	    }
	  if (!_train)
	    {
	      if (!_db_fname.empty())
		{
		  _test_dbfullname = _db_fname;
		  _db = true;
		  return; // done
		}
	      _svmdata_test = std::move(_svmdata);
	    }
	  else _svmdata.clear();
	  int n = 0;
	  auto hit = _svmdata_test.begin();
	  while(hit!=_svmdata_test.end())
	    {
	      _dv_test_sparse.push_back(to_sparse_datum((*hit)));
	      if (!_train)
		_ids.push_back(std::to_string(n));
	      ++n;
	      ++hit;
	    }
	}
    }

    caffe::SparseDatum to_sparse_datum(const SVMline &svml)
      {
	caffe::SparseDatum datum;
	datum.set_label(svml._label);
	auto hit = svml._v.begin();
	int nelts = 0;
	while(hit!=svml._v.end())
	  {
	    datum.add_data(static_cast<float>((*hit).second));
	    datum.add_indices((*hit).first);
	    ++nelts;
	    ++hit;
	  }
	datum.set_nnz(nelts);
	datum.set_size(channels());
	return datum;
      }

    std::vector<caffe::SparseDatum> get_dv_test_sparse_db(const int &num);
    std::vector<caffe::SparseDatum> get_dv_test_sparse(const int &num)
      {
	if (!_db)
	  {
	    int i = 0;
	    std::vector<caffe::SparseDatum> dv;
	    while(_dt_vit!=_dv_test_sparse.end()
		  && i < num)
	      {
		dv.push_back((*_dt_vit));
		++i;
		++_dt_vit;
	      }
	    return dv;
	  }
	  else return get_dv_test_sparse_db(num);
      }

    void reset_dv_test();

  private:
    int svm_to_db(const std::string &traindbname,
		  const std::string &testdbname,
		  const APIData &ad_input,
		  const std::string &backend="lmdb"); // lmdb, leveldb

    void write_svmline_to_db(const std::string &dbfullname,
			     const std::string &testdbfullname,
			     const APIData &ad_input,
			     const std::string &backend="lmdb");

  public:
    std::vector<caffe::SparseDatum>::const_iterator _dt_vit;
    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
 
  private:
    std::unique_ptr<caffe::db::Transaction> _txn;
    std::unique_ptr<caffe::db::DB> _tdb;
    std::unique_ptr<caffe::db::Transaction> _ttxn;
    std::unique_ptr<caffe::db::DB> _ttdb;
    int _channels = 0;
  };
  
}

#endif
