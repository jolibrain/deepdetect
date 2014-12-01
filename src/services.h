/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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
  typedef mapbox::util::variant<MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>> mls_variant_type;
  
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

  class visitor_train : public mapbox::util::static_visitor<output>
  {
  public:
    visitor_train() {}
    ~visitor_train() {}
    
    template<typename T>
      output operator() (T &mllib)
      {
        int r = mllib.train(_ad,_out);
	return output(r,_out);
      }
    
    APIData _ad;
    APIData _out;
  };
  
  class Services
  {
  public:
    Services() {};
    ~Services() {};

    size_t services_size() const
    {
      return _mlservices.size();
    }
    
    void add_service(const std::string &sname,
		     mls_variant_type &&mls) 
    {
      std::lock_guard<std::mutex> lock(_mlservices_mtx);
      _mlservices.push_back(std::move(mls));
      _mlservidx.insert(std::pair<std::string,int>(sname,_mlservices.size()-1));
    }

    bool remove_service(const std::string &sname)
    {
      std::lock_guard<std::mutex> lock(_mlservices_mtx);
      auto hit = _mlservidx.begin();
      if ((hit=_mlservidx.find(sname))!=_mlservidx.end())
	{
	  _mlservices.erase(_mlservices.begin()+(*hit).second);
	  _mlservidx.erase(hit);
	  return true;
	}
      LOG(ERROR) << "cannot find service " << sname << " for removal\n";
      return false;
    }

    int get_service_pos(const std::string &sname)
    {
       std::lock_guard<std::mutex> lock(_mlservices_mtx);
       std::unordered_map<std::string,int>::const_iterator hit;
       if ((hit=_mlservidx.find(sname))!=_mlservidx.end())
	 return (*hit).second;
       else return -1;
    }

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
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " training call failed\n";
	}
      out = pout._out;
      std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
      out.add("time",elapsed);
      return pout._status;
    }
    
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
      catch(...)
	{
	  LOG(ERROR) << "service #" << pos << " prediction call failed\n";
	}
      out = pout._out;
      std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
      out.add("time",elapsed);
      return pout._status;
    }

    std::vector<mls_variant_type> _mlservices;
    std::unordered_map<std::string,int> _mlservidx;
    
    std::mutex _mlservices_mtx; /**< mutex around adding/removing services. */
  };
  
}

#endif
