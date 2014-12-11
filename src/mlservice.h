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

#ifndef MLSERVICE_H
#define MLSERVICE_H

#include "mllibstrategy.h"
#include "mlmodel.h"
#include <string>
#include <future>
#include <mutex>
#include <unordered_map>
#include <iostream>

namespace dd
{
  template<template <class U,class V,class W> class TMLLib, class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    class MLService : public TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
  {
  public:
    MLService(const std::string &sname,
	      const TMLModel &mlmodel,
	      const std::string &description="")
      :TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>(mlmodel),_sname(sname),_description(description)
      {}
    MLService(MLService &&mls) noexcept
      :TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>(std::move(mls)),_sname(std::move(mls._sname)),_description(std::move(mls._description))
      {}
    ~MLService() {}

    APIData info() const
    {
      APIData ad;
      ad.add("name",_sname);
      ad.add("description",_description);
      ad.add("mllib",this->_libname);
      return ad;
    }
    
    // To be surcharged in related classes
    APIData status() const
    {
      APIData ad;
      ad.add("name",_sname);
      ad.add("description",_description);
      ad.add("mllib",this->_libname);
      return ad;
    }

    int train_job(const APIData &ad, APIData &out)
    {
      if (ad.has("async") && ad.get("async").get<bool>())
	{
	  std::lock_guard<std::mutex> lock(_tjobs_mutex);
	  _training_jobs.emplace(++_tjobs_counter,
				 std::async(std::launch::async,
					    [this,ad,&out]{ return this->train(ad,out); }));
	  std::cout << "launched training job\n";
	  return _tjobs_counter;
	}
	else return this->train(ad,out);
    }

    int training_job_status(const APIData &ad, APIData &out)
    {
      int j = static_cast<int>(ad.get("job").get<double>());
      int secs = 1;
      if (ad.has("timeout"))
	secs = static_cast<int>(ad.get("timeout").get<double>());
      std::unordered_map<int,std::future<int>>::iterator hit;
      if ((hit=_training_jobs.find(j))!=_training_jobs.end())
	{
	  std::future_status status = (*hit).second.wait_for(std::chrono::seconds(secs));
	  if (status == std::future_status::timeout)
	    {
	      out.add("status","running");
	    }
	  else if (status == std::future_status::ready)
	    {
	      int st = (*hit).second.get(); //TODO: exception handling ?
	      out.add("status",st);
	    }
	  return 0;
	}
      else
	{
	  return 1;
	}
    }

    std::string _sname; /**< service name. */
    std::string _description; /**< optional description of the service. */
  
    int _tjobs_counter = 0; /**< training jobs counter. */
    std::mutex _tjobs_mutex;
    std::unordered_map<int,std::future<int>> _training_jobs; // XXX: the futures' dtor blocks if the object is being terminated
  };
  
}

#endif
