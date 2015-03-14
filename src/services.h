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

#ifndef SERVICES_H
#define SERVICES_H

#include "utils/variant.hpp"
#include "mlservice.h"
#include "apidata.h"
#include "inputconnectorstrategy.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"
#include "caffelib.h"
#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>

namespace dd
{
  /* service types as variant type. */
  typedef mapbox::util::variant<MLService<CaffeLib,ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>,
    MLService<CaffeLib,CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>> mls_variant_type;
  
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

  class visitor_predict : public mapbox::util::static_visitor<output>
  {
  public:
    visitor_predict() {}
    ~visitor_predict() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.predict(_ad,_out);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
  };

  /**
   * \brief training job visitor class
   */
  class visitor_train : public mapbox::util::static_visitor<output>
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
  class visitor_train_status : public mapbox::util::static_visitor<output>
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
  class visitor_train_delete : public mapbox::util::static_visitor<output>
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
  class visitor_init : public mapbox::util::static_visitor<>
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
  class visitor_clear : public mapbox::util::static_visitor<>
  {
  public:
    visitor_clear(const APIData &ad)
      :_ad(ad) {}
    ~visitor_clear() {}
    
    template<typename T>
      void operator() (T &mllib)
      {
	if (_ad.has("clear"))
	  {
	    std::string clear = _ad.get("clear").get<std::string>();
	    if (clear == "full")
	      mllib.clear_full();
	    else if (clear == "lib")
	      mllib.clear_mllib(_ad);
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
      visitor_init vi(ad);
      try
	{
	  mapbox::util::apply_visitor(vi,mls);
	  std::lock_guard<std::mutex> lock(_mlservices_mtx);
	  _mlservices.push_back(std::move(mls));
	  _mlservidx.insert(std::pair<std::string,int>(sname,_mlservices.size()-1));
	}
      catch (InputConnectorBadParamException &e)
	{
	  LOG(ERROR) << "service creation input connector bad param: " << e.what() << std::endl;
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  LOG(ERROR) << "service creation mllib bad param: " << e.what() << std::endl;
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  LOG(ERROR) << "service creation mllib internal error: " << e.what() << std::endl;
	  throw;
	}
      catch(...)
	{
	  LOG(ERROR) << "service creation call failed\n";
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
      auto hit = _mlservidx.begin();
      if ((hit=_mlservidx.find(sname))!=_mlservidx.end())
	{
	  if (ad.has("clear"))
	    {
	      visitor_clear vc(ad);
	      try
		{
		  mapbox::util::apply_visitor(vc,_mlservices.at((*hit).second));
		}
	      catch (MLLibBadParamException &e)
		{
		  LOG(ERROR) << "service " << sname << " mllib bad param: " << e.what() << std::endl;
		  throw;
		}
	      catch (MLLibInternalException &e)
		{
		  LOG(ERROR) << "service " << sname << " mllib internal error: " << e.what() << std::endl;
	  	  throw;
		}
	      catch(...)
		{
		  LOG(ERROR) << "service " << sname << " delete service call failed\n";
	  	  throw;
		}
	    }
	  _mlservices.erase(_mlservices.begin()+(*hit).second);
	  _mlservidx.erase(hit);
	  return true;
	}
      LOG(ERROR) << "cannot find service " << sname << " for removal\n";
      return false;
    }

    /**
     * \brief get a service's position in the services container
     * @return service position, -1 if not found
     */
    int get_service_pos(const std::string &sname)
    {
       std::lock_guard<std::mutex> lock(_mlservices_mtx);
       std::unordered_map<std::string,int>::const_iterator hit;
       if ((hit=_mlservidx.find(sname))!=_mlservidx.end())
	 return (*hit).second;
       else return -1;
    }

    /**
     * \brief train a statistical model using a service
     * @param ad root data object
     * @param pos service position
     * @param out output data object
     */
    int train(const APIData &ad, const int &pos, APIData &out)
    {
      std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
      visitor_train vt;
      vt._ad = ad;
      output pout;
      try
	{
	  pout = mapbox::util::apply_visitor(vt,_mlservices.at(pos));
	}
      catch (InputConnectorBadParamException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib bad param: " << e.what() << std::endl;
	  pout._status = -2;
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib bad param: " << e.what() << std::endl;
	  pout._status = -2;
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib internal error: " << e.what() << std::endl;
	  pout._status = -1;
	  throw;
	}
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " training call failed\n";
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      if (ad.has("async") && ad.get("async").get<bool>())
	{
	  //TODO: beware out is a ref that might be lost
	  out.add("job",pout._status); // status holds the job id...
	  out.add("status","running");
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
     * @param pos service position
     * @param out output data object
     */
    int train_status(const APIData &ad, const int &pos, APIData &out)
    {
      visitor_train_status vt;
      vt._ad = ad;
      output pout;
      try
	{
	  pout = mapbox::util::apply_visitor(vt,_mlservices.at(pos));
	}
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " training status call failed\n";
	  pout._status = -1;
	}
      out = pout._out;
      return pout._status;
    }
    
    /**
     * \brief kills a training job
     * @param ad root data object
     * @param pos service position
     * @param output data object
     */
    int train_delete(const APIData &ad, const int &pos, APIData &out)
    {
      visitor_train_delete vt;
      vt._ad = ad;
      output pout;
      try
	{
	  pout = mapbox::util::apply_visitor(vt,_mlservices.at(pos));
	}
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " training delete call failed\n";
	  pout._status = -1;
	}
      out = pout._out;
      return pout._status;
    }
    
    /**
     * \brief prediction from statistical model
     * @param ad root data object
     * @param pos service position
     * @param out output data object
     */
    int predict(const APIData &ad, const int &pos, APIData &out)
    {
      std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
      visitor_predict vp;
      vp._ad = ad;
      output pout;
      try
	{
	  pout = mapbox::util::apply_visitor(vp,_mlservices.at(pos));
	}
      catch (InputConnectorBadParamException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib bad param: " << e.what() << std::endl;
	  pout._status = -2;
	  throw;
	}
      catch (MLLibBadParamException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib bad param: " << e.what() << std::endl;
	  pout._status = -2;
	  throw;
	}
      catch (MLLibInternalException &e)
	{
	  LOG(ERROR) << "service #" << pos << " mllib internal error: " << e.what() << std::endl;
	  pout._status = -1;
	  throw;
	}
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " prediction call failed\n";
	  pout._status = -1;
	  throw;
	}
      out = pout._out;
      std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
      out.add("time",elapsed);
      return pout._status;
    }

    std::vector<mls_variant_type> _mlservices; /**< container of instanciated services. */
    std::unordered_map<std::string,int> _mlservidx; /**< services position per service name. */
    
  protected:
    std::mutex _mlservices_mtx; /**< mutex around adding/removing services. */
  };
  
}

#endif
