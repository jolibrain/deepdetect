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

#ifndef CAFFEINPUTCONNS_H
#define CAFFEINPUTCONNS_H

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "caffe/caffe.hpp"
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
      :_dv(cii._dv),_dv_test(cii._dv_test),_ids(cii._ids),_test_labels(cii._test_labels) {}
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

    void reset_dv_test() {}
    
    std::vector<caffe::Datum> _dv; /**< main input datum vector, used for training or prediction */
    std::vector<caffe::Datum> _dv_test; /**< test input datum vector, when applicable in training mode */
    std::vector<std::string> _ids; /**< input ids (e.g. image ids) */
    std::vector<int> _test_labels; /**< vector of test labels, kept aside for measuring net output. */
  };

  /**
   * \brief Caffe image connector, supports both files and building of database for training
   */
  class ImgCaffeInputFileConn : public ImgInputFileConn, public CaffeInputInterface
  {
  public:
    ImgCaffeInputFileConn()
      :ImgInputFileConn() {}
    ImgCaffeInputFileConn(const ImgCaffeInputFileConn &i)
      :ImgInputFileConn(i),CaffeInputInterface(i) {}
    ~ImgCaffeInputFileConn() 
      {
	//if (_test_db)
	//_test_db->Close();
      }

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
    }

    void transform(const APIData &ad)
    {
      // in prediction mode, convert the images to Datum, a Caffe data structure
      if (!_train)
	{
	  try
	    {
	      ImgInputFileConn::transform(ad);
	    }
	  catch (InputConnectorBadParamException &e)
	    {
	      if (!(_train && _uris.empty())) // Caffe model files can reference the source to the image training data 
		throw;
	    }
	  for (int i=0;i<(int)this->_images.size();i++)
	    {      
	      caffe::Datum datum;
	      caffe::CVMatToDatum(this->_images.at(i),&datum);
	      _dv.push_back(datum);
	      _ids.push_back(this->_uris.at(i));
	    }
	}
      else // more complicated, since images can be heavy, a db is built so that it is less costly to iterate than the filesystem
	{
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
	  if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	    {
	      APIData ad_param = ad.getobj("parameters");
	      if (ad_param.has("input"))
		{
		  fillup_parameters(ad_param.getobj("input"));
		}
	    }
	  
	  // create db
	  _model_repo = ad.get("model_repo").get<std::string>();
	  images_to_db(_uris.at(0),_model_repo + "/" + _dbname,_model_repo + "/" + _test_dbname);
	  
	  // compute mean of images, not forcely used, depends on net, see has_mean_file
	  compute_images_mean(_model_repo + "/" + _dbname,
			      _model_repo + "/" + _meanfname);

	  // enrich data object with db files location
	  APIData dbad;
	  dbad.add("train_db",_model_repo + "/" + _dbfullname);
	  if (_test_split > 0.0)
	    dbad.add("test_db",_model_repo + "/" + _test_dbfullname);
	  dbad.add("meanfile",_model_repo + "/" + _meanfname);
	  std::vector<APIData> vdbad = {dbad};
	  const_cast<APIData&>(ad).add("db",vdbad);
	}
    }

    std::vector<caffe::Datum> get_dv_test(const int &num,
					  const bool &has_mean_file);

    void reset_dv_test();
    
  private:
    int images_to_db(const std::string &rfolder,
		     const std::string &traindbname,
		     const std::string &testdbname,
		     const std::string &backend="lmdb", // lmdb, leveldb
		     const bool &encoded=false, // save the encoded image in datum
		     const std::string &encode_type=""); // 'png', 'jpg', ...

    void write_image_to_db(const std::string &dbfullname,
			   const std::vector<std::pair<std::string,int>> &lfiles,
			   std::vector<int> &datum_labels,
			   const std::string &backend,
			   const bool &encoded,
			   const std::string &encode_type);
    
    int compute_images_mean(const std::string &dbname,
			    const std::string &meanfile,
			    const std::string &backend="lmdb");

  public:
    int _db_batchsize = -1;
    int _db_testbatchsize = -1;
    std::unique_ptr<caffe::db::DB> _test_db;
    std::unique_ptr<caffe::db::Cursor> _test_db_cursor;
    std::string _dbname = "train";
    std::string _test_dbname = "test";
    std::string _dbfullname = "train.lmdb";
    std::string _test_dbfullname = "test.lmdb";
    std::string _meanfname = "mean.binaryproto";
    std::string _model_repo;
    std::string _correspname = "corresp.txt";
  };

  /**
   * \brief Caffe CSV connector
   * \note use 'label_offset' in API to make sure that labels start at 0
   */
  class CSVCaffeInputFileConn : public CSVInputFileConn, public CaffeInputInterface
  {
  public:
    CSVCaffeInputFileConn()
      :CSVInputFileConn() 
      {
	reset_dv_test();
      }
    ~CSVCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }

    // size of each element in Caffe jargon
    int channels() const
    {
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
    
    void transform(const APIData &ad)
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
      auto hit = _csvdata.cbegin();
      while(hit!=_csvdata.cend())
	{
	  _dv.push_back(to_datum((*hit)._v));
	  _ids.push_back((*hit)._str);
	  ++hit;
	}
      _csvdata.clear();
      hit = _csvdata_test.cbegin();
      while(hit!=_csvdata_test.cend())
	{
	  // no ids taken on the test set
	  caffe::Datum dat = to_datum((*hit)._v);
	  _dv_test.push_back(dat);
	  _test_labels.push_back(dat.label());
	  ++hit;
	}
      _csvdata_test.clear();
    }

    std::vector<caffe::Datum> get_dv_test(const int &num,
					  const bool &has_mean_file)
      {
	(void)has_mean_file;
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

    void reset_dv_test()
    {
      _dt_vit = _dv_test.begin();
    }

    /**
     * \brief turns a vector of values into a Caffe Datum structure
     * @param vector of values
     * @return datum
     */
    caffe::Datum to_datum(const std::vector<double> &vf)
    {
      caffe::Datum datum;
      int datum_channels = vf.size();
      if (!_label.empty())
	datum_channels--;
      if (!_id.empty())
	datum_channels--;
      datum.set_channels(datum_channels);
      datum.set_height(1);
      datum.set_width(1);
      auto lit = _columns.begin();
      for (int i=0;i<(int)vf.size();i++)
	{
	  if (i == _label_pos)
	    {
	      datum.set_label(static_cast<int>(vf.at(i)+this->_label_offset));
	    }
	  else if (i == _id_pos)
	    {
	      ++lit;
	      continue;
	    }
	  else 
	    {
	      datum.add_float_data(static_cast<float>(vf.at(i)));
	    }
	  ++lit;
	}
      return datum;
    }

    std::vector<caffe::Datum>::const_iterator _dt_vit;
  };

}

#endif
