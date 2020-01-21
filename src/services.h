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

#ifndef SERVICES_H
#define SERVICES_H

#include "utils/variant.hpp"
#include "mlservice.h"
#include "apidata.h"
#include "inputconnectorstrategy.h"
#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"
#include "svminputfileconn.h"
#include "outputconnectorstrategy.h"
#include "chain.h"
#include "chain_actions.h"
#ifdef USE_CAFFE
#include "backends/caffe/caffelib.h"
#endif
#ifdef USE_TF
#include "backends/tf/tflib.h"
#endif
#ifdef USE_DLIB
#include "backends/dlib/dliblib.h"
#endif
#ifdef USE_CAFFE2
#include "backends/caffe2/caffe2lib.h"
#endif
#ifdef USE_XGBOOST
#include "backends/xgb/xgblib.h"
#endif
#ifdef USE_TSNE
#include "backends/tsne/tsnelib.h"
#endif
#ifdef USE_NCNN
#include "backends/ncnn/ncnnlib.h"
#endif
#ifdef USE_TORCH
#include "backends/torch/torchlib.h"
#endif
#ifdef USE_TENSORRT
#include "backends/tensorrt/tensorrtlib.h"
#endif
#include <spdlog/spdlog.h>
#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>

namespace dd
{
  /* service types as variant type. */
  typedef mapbox::util::variant<
#ifdef USE_CAFFE
    MLService<CaffeLib,ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,CSVTSCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,SVMCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,ImgCaffeInputFileConn,UnsupervisedOutput,CaffeModel>,
    MLService<CaffeLib,CSVCaffeInputFileConn,UnsupervisedOutput,CaffeModel>,
    MLService<CaffeLib,CSVTSCaffeInputFileConn,UnsupervisedOutput,CaffeModel>,
    MLService<CaffeLib,TxtCaffeInputFileConn,UnsupervisedOutput,CaffeModel>,
    MLService<CaffeLib,SVMCaffeInputFileConn,UnsupervisedOutput,CaffeModel>
#endif
#ifdef USE_CAFFE2
    #ifdef USE_CAFFE
    ,
    #endif
    MLService<Caffe2Lib,ImgCaffe2InputFileConn,SupervisedOutput,Caffe2Model>,
    MLService<Caffe2Lib,ImgCaffe2InputFileConn,UnsupervisedOutput,Caffe2Model>
#endif
#ifdef USE_TF
    #if defined(USE_CAFFE) || defined(USE_CAFFE2)
    ,
    #endif
    MLService<TFLib,ImgTFInputFileConn,SupervisedOutput,TFModel>,
    MLService<TFLib,ImgTFInputFileConn,UnsupervisedOutput,TFModel>
#endif
#ifdef USE_DLIB
    #if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF)
    ,
    #endif
    MLService<DlibLib,ImgDlibInputFileConn,SupervisedOutput,DlibModel>,
    MLService<DlibLib,ImgDlibInputFileConn,UnsupervisedOutput,DlibModel>
#endif
#ifdef USE_XGBOOST
    #if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF) || defined(USE_DLIB)
    ,
    #endif
    MLService<XGBLib,CSVXGBInputFileConn,SupervisedOutput,XGBModel>,
    MLService<XGBLib,SVMXGBInputFileConn,SupervisedOutput,XGBModel>,
    MLService<XGBLib,TxtXGBInputFileConn,SupervisedOutput,XGBModel>
#endif
#ifdef USE_TSNE
    #if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF) || defined(USE_DLIB) || defined(USE_XGBOOST)
    ,
    #endif
    MLService<TSNELib,CSVTSNEInputFileConn,UnsupervisedOutput,TSNEModel>,
    MLService<TSNELib,TxtTSNEInputFileConn,UnsupervisedOutput,TSNEModel>
#endif
#ifdef USE_NCNN
    #if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF) || defined(USE_DLIB) || defined(USE_XGBOOST) || defined(USE_TSNE)
    ,
    #endif
    MLService<NCNNLib,CSVTSNCNNInputFileConn,SupervisedOutput,NCNNModel>,
    MLService<NCNNLib,ImgNCNNInputFileConn,SupervisedOutput,NCNNModel>
#endif
#ifdef USE_TORCH
    #if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF) || defined(USE_DLIB) || defined(USE_XGBOOST) || defined(USE_TSNE) || defined(USE_NCNN)
    ,
    #endif
    MLService<TorchLib,ImgTorchInputFileConn,SupervisedOutput,TorchModel>,
    MLService<TorchLib,TxtTorchInputFileConn,SupervisedOutput,TorchModel>
#endif
#ifdef USE_TENSORRT
#if defined(USE_CAFFE) || defined(USE_CAFFE2) || defined(USE_TF) || defined(USE_DLIB) || defined(USE_XGBOOST) || defined(USE_TSNE) || defined(USE_NCNN) || defined(USE_TORCH)
    ,
#endif
    MLService<TensorRTLib,ImgTensorRTInputFileConn,SupervisedOutput,TensorRTModel>
#endif
    > mls_variant_type;

  class ServiceForbiddenException : public std::exception
  {
  public:
    ServiceForbiddenException(const std::string &s)
      :_s(s) {}
    ~ServiceForbiddenException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  class ServiceNotFoundException : public std::exception
  {
  public:
    ServiceNotFoundException(const std::string &s)
      :_s(s) {}
    ~ServiceNotFoundException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };
  
  class output
  {
  public:
    output() {}
    output(const int &status, const APIData &out)
      :_status(status),_out(out)
    {}
    ~output() 
      {}
    
    int _status = 0;
    APIData _out;
  };

  class visitor_predict
  {
  public:
    visitor_predict() {}
    ~visitor_predict() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.predict_job(_ad,_out,_chain);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
    bool _chain = false;
  };

  /**
   * \brief training job visitor class
   */
  class visitor_train
  {
  public:
    visitor_train() {}
    ~visitor_train() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.train_job(_ad,_out);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
  };

  /**
   * \brief training job status visitor class
   */
  class visitor_train_status
  {
  public:
    visitor_train_status() {}
    ~visitor_train_status() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.training_job_status(_ad,_out);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
  };

  /**
   * \brief training job deletion class
   */
  class visitor_train_delete
  {
  public:
    visitor_train_delete() {}
    ~visitor_train_delete() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.training_job_delete(_ad,_out);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
  };

  /**
   * \brief service initialization visitor class
   */
  class visitor_init
  {
  public:
    visitor_init(const APIData &ad)
      :_ad(ad) {}
    ~visitor_init() {}
    
    template<typename T>
      void operator() (T &mllib)
      {
	mllib.init(_ad);
      }
    
    APIData _ad;
  };
  
  /**
   * \brief service deletion visitor class
   */
  class visitor_clear
  {
  public:
    visitor_clear(const APIData &ad)
      :_ad(ad) {}
    ~visitor_clear() {}
    
    template<typename T>
      void operator() (T &mllib)
      {
	mllib.kill_jobs();
	if (_ad.has("clear"))
	  {
	    std::string clear = _ad.get("clear").get<std::string>();
	    if (clear == "full")
	      mllib.clear_full();
#ifdef USE_SIMSEARCH
	    else if (clear == "lib")
	      {
		mllib.clear_mllib(_ad);
		mllib.clear_index();
	      }
#else
	    else if (clear == "lib")
	      mllib.clear_mllib(_ad);
#endif
           else if (clear == "dir")
             mllib.clear_dir();
#ifdef USE_SIMSEARCH
	    else if (clear == "index")
	      mllib.clear_index();
#endif
	  }
      }
    
    APIData _ad;
  };
  
  /**
   * \brief class for deepetect machine learning services.
   *        Each service instanciates a machine learning library and channels
   *        data for training and prediction along with parameters from API
   *        Service uses a variant type and store instances in a single iterable container.
   */
  class Services
  {
  public:
    Services() {}
    ~Services() {}

    /**
     * \brief get number of services
     * @return number of service instances
     */
    size_t services_size() const
    {
      return _mlservices.size();
    }
    
    /**
     * \brief add a new service
     * @param sname service name
     * @param mls service object as variant
     * @param ad optional root data object holding service's parameters
     */
    void add_service(const std::string &sname,
		     mls_variant_type &&mls,
		     const APIData &ad=APIData()) 
    {
      std::unordered_map<std::string,mls_variant_type>::const_iterator hit;
      if ((hit=_mlservices.find(sname))!=_mlservices.end())
	{
	  throw ServiceForbiddenException("Service already exists");
	}

      auto llog = spdlog::get(sname);
      visitor_init vi(ad);
      try
	{
	  mapbox::util::apply_visitor(vi,mls);
	  std::lock_guard<std::mutex> lock(_mlservices_mtx);
	  _mlservices.insert(std::pair<std::string,mls_variant_type>(sname,std::move(mls)));
	}
      catch (InputConnectorBadParamException &e)
	{
	  llog->error("service creation input connector bad param: {}",e.what());
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  llog->error("service creation mllib bad param: {}",e.what());
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  llog->error("service creation mllib internal error: {}",e.what());
	  throw;
	}
      catch(...)
	{
	  llog->error("service creation call failed");
	  throw;
	}
    }

    /**
     * \brief removes and destroys a service
     * @param sname service name
     * @param ad root data object
     * @return true if service was removed, false otherwise (i.e. not found)
     */
    bool remove_service(const std::string &sname,
			const APIData &ad)
    {
      std::lock_guard<std::mutex> lock(_mlservices_mtx);
      auto hit = _mlservices.begin();
      if ((hit=_mlservices.find(sname))!=_mlservices.end())
	{
	  auto llog = spdlog::get(sname);
	  if (ad.has("clear"))
	    {
	      visitor_clear vc(ad);
	      try
		{
		  mapbox::util::apply_visitor(vc,(*hit).second);
		}
	      catch (MLLibBadParamException &e)
		{
		  llog->error("mllib bad param: {}",e.what());
		  throw;
		}
	      catch (MLLibInternalException &e)
		{
		  llog->error("mllib internal error: {}",e.what());
	  	  throw;
		}
	      catch(...)
		{
		  llog->error("delete service call failed");
	  	  throw;
		}
	    }
	  _mlservices.erase(hit);
	  return true;
	}
      auto llog = spdlog::get("api");
      llog->error("cannot find service for removal");
      return false;
    }

    /**
     * \brief get a service position as iterator
     * @param sname service name
     * @return service position, end of container if not found
     */
    std::unordered_map<std::string,mls_variant_type>::iterator get_service_it(const std::string &sname)
      {
	std::unordered_map<std::string,mls_variant_type>::iterator hit;
	if ((hit=_mlservices.find(sname))!=_mlservices.end())
	  return hit;
	return _mlservices.end();
      }

    /**
     * \brief checks whether a service exists
     * @param sname service name
     * return true if service exists, false otherwise
     */
    bool service_exists(const std::string &sname)
    {
      auto hit = get_service_it(sname);
      if (hit == _mlservices.end())
	return false;
      return true;
    }
    
    /**
     * \brief train a statistical model using a service
     * @param ad root data object
     * @param sname service name
     * @param out output data object
     */
    int train(const APIData &ad, const std::string &sname, APIData &out)
    {
      std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
      visitor_train vt;
      vt._ad = ad;
      output pout;
      auto llog = spdlog::get(sname);
      try
	{
	  auto hit = get_service_it(sname);
	  pout = mapbox::util::apply_visitor(vt,(*hit).second);
	}
      catch (InputConnectorBadParamException &e)
	{
	  llog->error("mllib bad param: {}",e.what());
	  pout._status = -2;
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  llog->error("mllib bad param: {}",e.what());
	  pout._status = -2;
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  llog->error("mllib internal error: {}",e.what());
	  pout._status = -1;
	  throw;
	}
      catch(...)
	{
	  llog->error("training call failed");
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      if (ad.has("async") && ad.get("async").get<bool>())
	{
	  out.add("job",pout._status); // status holds the job id...
	  out.add("status",std::string("running"));
	  return 0; // status is OK i.e. the job has started.
	}
      else
	{
	  std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
	  double elapsed = std::chrono::duration_cast<std::chrono::seconds>(tstop-tstart).count();
	  out.add("time",elapsed);
	}
      return pout._status;
    }
    
    /**
     * \brief access to training job status
     * @param ad root data object
     * @param sname service name
     * @param out output data object
     */
    int train_status(const APIData &ad, const std::string &sname, APIData &out)
    {
      visitor_train_status vt;
      vt._ad = ad;
      output pout;
      try
	{
	  auto hit = get_service_it(sname);
	  pout = mapbox::util::apply_visitor(vt,(*hit).second);
	}
      catch(...)
	{
	  auto llog = spdlog::get(sname);
	  llog->error("training status call failed");
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      return pout._status;
    }
    
    /**
     * \brief kills a training job
     * @param ad root data object
     * @param sname service name
     * @param output data object
     */
    int train_delete(const APIData &ad, const std::string &sname, APIData &out)
    {
      visitor_train_delete vt;
      vt._ad = ad;
      output pout;
      try
	{
	  auto hit = get_service_it(sname);
	  pout = mapbox::util::apply_visitor(vt,(*hit).second);
	}
      catch(...)
	{
	  auto llog = spdlog::get(sname);
	  llog->error("training delete call failed");
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      return pout._status;
    }
    
    /**
     * \brief prediction from statistical model
     * @param ad root data object
     * @param sname service name
     * @param out output data object
     */
    int predict(const APIData &ad, const std::string &sname, APIData &out, const bool &chain=false)
    {
      std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
      visitor_predict vp;
      vp._ad = ad;
      vp._chain = chain;
      output pout;
      auto llog = spdlog::get(sname);
      try
	{
	  auto hit = get_service_it(sname);
	  pout = mapbox::util::apply_visitor(vp,(*hit).second);
	}
      catch (InputConnectorBadParamException &e)
	{
	  llog->error("mllib bad param: {}",e.what());
	  pout._status = -2;
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  llog->error("mllib bad param: {}",e.what());
	  pout._status = -2;
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  llog->error("mllib internal error: {}",e.what());
	  pout._status = -1;
	  throw;
	}
      catch (MLServiceLockException &e)
	{
	  llog->error("mllib lock error: {}",e.what());
	  pout._status = -3;
	  throw;
	}
	  catch (const std::exception &e)
    {
      // catch anything thrown within try block that derives from std::exception
	  llog->error("other error: {}",e.what());
	  pout._status = -1;
	  throw;
	}
      catch(...)
	{
	  llog->error("prediction call failed");
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
      out.add("time",elapsed);
      return pout._status;
    }

    int chain_service(const std::string &cname,
		      const std::shared_ptr<spdlog::logger> &chain_logger,
		      APIData &adc,
		      ChainData &cdata,
		      const std::string &pred_id,
		      std::vector<std::string> &meta_uris,
		      std::vector<std::string> &index_uris,
		      const std::string &parent_id,
		      const int chain_pos,
		      int &npredicts)
    {
      std::string sname = adc.get("service").get<std::string>();
      chain_logger->info("[" + std::to_string(chain_pos) + "] / executing predict on service " + sname);
      
      // need to check that service exists
      if (!service_exists(sname))
	{
	  spdlog::drop(cname);
	  throw ServiceNotFoundException("Service " + sname + " does not exist");
	}

      // if not first predict call in the chain, need to setup the input data!
      if (chain_pos != 0)
	{
	  // take data from the previous action
	  APIData act_data = cdata.get_action_data(parent_id);
	  if (act_data.empty())
	    {
	      spdlog::drop(cname);
	      throw InputConnectorBadParamException("no action data for action id " + parent_id);
	    }
	  if (act_data.has("data"))
	    {
	      adc.add("data",act_data.get("data").get<std::vector<std::string>>()); // action output data must be string for now (more types to be supported / auto-detected)
	    }
	  else if (act_data.has("data_raw_img")) // raw images
	    {
	      adc.add("data_raw_img",act_data.get("data_raw_img").get<std::vector<cv::Mat>>());
	    }
	  adc.add("ids",act_data.get("cids").get<std::vector<std::string>>()); // chain ids of processed elements
	  adc.add("meta_uris",meta_uris);
	  adc.add("index_uris",index_uris);
	}
      else {
	cdata._first_id = pred_id;
      }
      
      APIData pred_out;
      try
	{
	  predict(adc,sname,pred_out,true);
	}
      catch(...)
	{
	  spdlog::drop(cname);
	  throw;
	}
      
      // check on results
      std::vector<APIData> vad = pred_out.getv("predictions");
      if (vad.empty())
	{
	  chain_logger->info("[" + std::to_string(chain_pos) + "]  no predictions");
	  return 1;
	}
      
      int classes_size = 0;
      int vals_size = 0;
      std::vector<std::string> nmeta_uris;
      std::vector<std::string> nindex_uris;
      for (size_t j=0;j<vad.size();j++)
	{
	  size_t npred_classes = vad.at(j).getv("classes").size();
	  classes_size += npred_classes;
	  vals_size += static_cast<int>(vad.at(j).has("vals"));
	  if (chain_pos == 0) // first call's response contains uniformized top level URIs.
	    {
	      for (size_t k=0;k<npred_classes;k++)
		{
		  nmeta_uris.push_back(vad.at(j).get("uri").get<std::string>());
		  if (vad.at(j).has("index_uri"))
		    nindex_uris.push_back(vad.at(j).get("index_uri").get<std::string>());
		}
	    }
	  else // update meta uris to batch size at the current level of the chain
	    {
	      for (size_t k=0;k<npred_classes;k++)
		{
		  nmeta_uris.push_back(meta_uris.at(j));
		  if (!index_uris.empty())
		    nindex_uris.push_back(index_uris.at(j));
		}
	    }
	}
      meta_uris = nmeta_uris;
      index_uris = nindex_uris;
      
      if (!classes_size && !vals_size)
	{
	  chain_logger->info("[" + std::to_string(chain_pos) + "] / no result from prediction");
	  return 1;
	}
      ++npredicts;
      
      // store model output
      cdata.add_model_data(pred_id,pred_out);

      return 0;
    }

    int chain_action(const std::shared_ptr<spdlog::logger> &chain_logger,
		     APIData &adc,
		     ChainData &cdata,
		     const int &chain_pos,
		     const std::string &prec_pred_id)
    {
      std::string action_type = adc.getobj("action").get("type").get<std::string>();

      APIData prev_data = cdata.get_model_data(prec_pred_id);
      if (!prev_data.getv("predictions").size())
	{
	  // no prediction to work from
	  chain_logger->info("no prediction to act on");
	  return 1;
	}
      
      // call chain action factory
      chain_logger->info("[" + std::to_string(chain_pos) + "] / executing action " + action_type);
      ChainActionFactory caf(adc);
      caf.apply_action(action_type,
		       prev_data,
		       cdata,
		       chain_logger);
      
      // replace prev_data in cdata for prec_pred_id
      cdata.add_model_data(prec_pred_id,prev_data);
      
      std::vector<APIData> vad = prev_data.getv("predictions");
      if (vad.empty())
	{
	  // no prediction to work from
	  chain_logger->info("no prediction to act on after applying action " + action_type);
	  return 1;
	}
      
      int classes_size = 0;
      int vals_size = 0;
      for (size_t i=0;i<vad.size();i++)
	{
	  int npred_classes = vad.at(i).getv("classes").size();
	  classes_size += npred_classes;
	  vals_size += static_cast<int>(vad.at(i).has("vals"));
	}
      
      if (!classes_size && !vals_size)
	{
	  chain_logger->info("[" + std::to_string(chain_pos) + "] / no result after applying action " + action_type);
	  return 1;
	}      

      return 0;
    }
    
    int chain(const APIData &ad, const std::string &cname, APIData &out)
    {
      try
	{
#ifdef USE_DD_SYSLOG
	  auto chain_logger = spdlog::syslog_logger(cname);
#else
	  auto chain_logger = spdlog::stdout_logger_mt(cname);
#endif
	  
	  std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
	  
	  // - iterate chain of calls
	  // - if predict call, use the visitor to execute it
	  //      - this requires storing output into mlservice object / have a flag for telling the called service it's part of a chain
	  // - if action call, execute the generic code for it
	  std::vector<APIData> ad_calls = ad.getobj("chain").getv("calls");
	  chain_logger->info("number of calls=" + std::to_string(ad_calls.size()));
	  
	  // debug
	  /*std::vector<std::string> ckeys = ad.list_keys();
	    for (auto s: ckeys)
	    std::cerr << s << std::endl;*/
	  //debug
	  
	  ChainData cdata;
	  std::vector<std::string> meta_uris;
	  std::vector<std::string> index_uris;
	  std::unordered_map<std::string,std::vector<std::string>> um_meta_uris;
	  std::unordered_map<std::string,std::vector<std::string>> um_index_uris;
	  int npredicts = 0;
	  std::string prec_pred_id;
	  std::string prec_action_id;
	  int aid = 0;
	  for (size_t i=0;i<ad_calls.size();i++)
	    {
	      APIData adc = ad_calls.at(i);
	      if (adc.has("service"))
		{
		  std::string pred_id;
		  if (adc.has("id"))
		    pred_id = adc.get("id").get<std::string>();
		  else pred_id = std::to_string(i);
		  
		  std::string parent_id;
		  if (adc.has("parent_id"))
		    parent_id = adc.get("parent_id").get<std::string>();
		  else parent_id = prec_action_id;
		  
		  auto hit = um_meta_uris.find(parent_id);
		  if (hit!=um_meta_uris.end())
		    meta_uris = (*hit).second;
		  hit = um_index_uris.find(parent_id);
		  if (hit!=um_index_uris.end())
		    index_uris = (*hit).second;
		  cdata.add_model_sname(pred_id,adc.get("service").get<std::string>());
		  if (chain_service(cname,chain_logger,adc,cdata,
				    pred_id,meta_uris,index_uris,
				    parent_id,i,npredicts))
		    break;
		  prec_pred_id = pred_id;
		}
	      else if (adc.has("action"))
		{
		  if (chain_action(chain_logger,adc,cdata,i,prec_pred_id))
		    break;
		  if (adc.has("id"))
		    prec_action_id = adc.get("id").get<std::string>();
		  else prec_action_id = std::to_string(aid);
		  um_meta_uris.insert(std::pair<std::string,std::vector<std::string>>(prec_action_id,meta_uris));
		  um_index_uris.insert(std::pair<std::string,std::vector<std::string>>(prec_action_id,index_uris));
		  ++aid;
		}
	    }
	  
	  // producing a nested output
	  APIData nested_out;
	  if (npredicts > 1)
	    nested_out = cdata.nested_chain_output();
	  else nested_out = cdata.get_model_data(cdata._first_id);
	  
	  out = nested_out;
	  std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
	  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
	  out.add("time",elapsed);
	}
      catch(...)
	{
	  spdlog::drop(cname);
	  throw;
	}
      spdlog::drop(cname);
      
      return 0;
    }
    
    std::unordered_map<std::string,mls_variant_type> _mlservices; /**< container of instanciated services. */
    
  protected:
    std::mutex _mlservices_mtx; /**< mutex around adding/removing services. */
  };
  
}

#endif
