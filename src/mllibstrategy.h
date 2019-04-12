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

#ifndef MLLIBSTRATEGY_H
#define MLLIBSTRATEGY_H

#include "apidata.h"
#include "utils/fileops.hpp"
#include <spdlog/spdlog.h>
#include <atomic>
#include <exception>
#include <mutex>

namespace dd
{
  /**
   * \brief ML library bad parameter exception
   */
  class MLLibBadParamException : public std::exception
  {
  public:
    MLLibBadParamException(const std::string &s)
      :_s(s) {}
    ~MLLibBadParamException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };
  
  /**
   * \brief ML library internal error exception
   */
  class MLLibInternalException : public std::exception
  {
  public:
    MLLibInternalException(const std::string &s)
      :_s(s) {}
    ~MLLibInternalException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  /**
   * \brief main class for machine learning library encapsulation
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    class MLLib
  {
  public:
    /**
     * \brief constructor from model
     */
    MLLib(const TMLModel &mlmodel)
      :_mlmodel(mlmodel),_tjob_running(false) {}
    
    /**
     * \brief copy-constructor
     */
    MLLib(MLLib &&mll) noexcept
      :_inputc(mll._inputc),_outputc(mll._outputc),_mltype(mll._mltype),_mlmodel(mll._mlmodel),_meas(mll._meas),_meas_per_iter(mll._meas_per_iter),_tjob_running(mll._tjob_running.load()),_logger(mll._logger)
      {}
    
    /**
     * \brief destructor
     */
    ~MLLib() {}

    /**
     * \brief initializes ML lib
     * @param ad data object for "parameters/mllib"
     */
    void init_mllib(const APIData &ad);

    /**
     * \brief clear the lib service from local model files etc...
     * @param ad root data object
     */
    void clear_mllib(const APIData &ad);

#ifdef USE_SIMSEARCH
    /**
     * \brief removes search index from model repository
     */
    void clear_index()
    {
      _mlmodel.remove_index();
    }
#endif
    
    /**
     * \brief removes everything in model repository
     */
    void clear_full()
    {
      int err = fileops::clear_directory(_mlmodel._repo);
      if (err > 0)
	throw MLLibBadParamException("Failed opening directory " + _mlmodel._repo + " for deleting files within");
      else if (err < 0)
	throw MLLibInternalException("Failed deleting all files in directory " + _mlmodel._repo);
    }

    /**
     * \brief removes everything in model repository + rmdir
     */
    void clear_dir()
    {
      clear_full();
      int err = fileops::remove_dir(_mlmodel._repo);
      if (err < 0)
        throw MLLibBadParamException("unable to remove dir " + _mlmodel._repo);

    }


    /**
     * \brief train new model
     * @param ad root data object
     * @param out output data object (e.g. loss, ...)
     * @return 0 if OK, 1 otherwise
     */
    int train(const APIData &ad, APIData &out);

    /**
     * \brief predicts from model
     * @param ad root data object
     * @param out output data object (e.g. predictions, ...)
     * @return 0 if OK, 1 otherwise
     */
    int predict(const APIData &ad, APIData &out);
    
    /**
     * \brief ML library status
     */
    int status() const;
    
    /**
     * \brief clear all measures history
     */
    void clear_all_meas_per_iter()
    {
      std::lock_guard<std::mutex> lock(_meas_per_iter_mutex);
      _meas_per_iter.clear();
    }

    /**
     * \brief add value to measure history
     * @param meas measure name
     * @param l measure value
     */
    void add_meas_per_iter(const std::string &meas, const double &l)
    {
      std::lock_guard<std::mutex> lock(_meas_per_iter_mutex);
      auto hit = _meas_per_iter.find(meas);
      if (hit!=_meas_per_iter.end())
	{
	  (*hit).second.push_back(l);
	  if ((int)(*hit).second.size() >= _max_meas_points)
	    {
	      // resolution is halved
	      std::vector<double> vmeas_short;
	      vmeas_short.reserve(_max_meas_points/2);
	      int di = 0;
	      for (size_t j=0;j<(*hit).second.size();j+=2)
		vmeas_short.at(di++) = (*hit).second.at(j);
	      (*hit).second = vmeas_short;
	    }
	}
      else
	{
	  std::vector<double> vmeas = {l};
	  _meas_per_iter.insert(std::pair<std::string,std::vector<double>>(meas,vmeas));
	}
    }

    /**
     * \brief sub-samples measure history to fit a fixed number of points at max
     * @param hist measure history vector
     * @param npoints max number of output points
     */
    std::vector<double> subsample_hist(const std::vector<double> &hist,
				       const int &npoints) const
    {
      std::vector<double> sub_hist;
      sub_hist.reserve(npoints);
      int rpoints = std::ceil(hist.size() / npoints) + 1;
      for (size_t i=0;i<hist.size();i+=rpoints)
	sub_hist.push_back(hist.at(i));
      return sub_hist;
    }
    
    /**
     * \brief collect current measures history into a data object
     * @param ad api data object
     * @param npoints max number of output points, < 0 if unbounded
     */
    void collect_measures_history(APIData &ad,
				  const int &npoints=-1) const
    {
      APIData meas_hist;
      std::lock_guard<std::mutex> lock(_meas_per_iter_mutex);
      auto hit = _meas_per_iter.begin();
      while(hit!=_meas_per_iter.end())
	{
	  if (npoints > 0 && (int)(*hit).second.size() > npoints)
	    meas_hist.add((*hit).first+"_hist",subsample_hist((*hit).second,npoints));
	  else meas_hist.add((*hit).first+"_hist",(*hit).second);
	  ++hit;
	}
      ad.add("measure_hist",meas_hist);
    }

    /**
     * \brief fill up the in-memory metrics from values gathered
     *        from metrics.json file into the api data object
     * @param ad the api data object holding the values
     */
    void fillup_measures_history(const APIData &ad)
    {
      APIData ad_params = ad.getobj("parameters");
      if (!ad_params.has("metrics"))
	return;
      APIData ad_metrics = ad_params.getobj("metrics");
      std::vector<std::string> mkeys = ad_metrics.list_keys();
      for (auto s: mkeys)
	{
	  std::vector<double> mdata = ad_metrics.get(s).get<std::vector<double>>();
	  s.replace(s.find("_hist"),5,"");
	  _meas_per_iter.insert(std::pair<std::string,std::vector<double>>(s,mdata));
	}
    }
    
    /**
     * \brief sets current value of a measure
     * @param meas measure name
     * @param l measure value
     */
    void add_meas(const std::string &meas, const double &l)
    {
      std::lock_guard<std::mutex> lock(_meas_mutex);
      auto hit = _meas.find(meas);
      if (hit!=_meas.end())
	(*hit).second = l;
      else _meas.insert(std::pair<std::string,double>(meas,l));
    }

    void add_meas(const std::string &meas, const std::vector<double> &vl,
		  const std::vector<std::string> &cnames)
    {
      std::lock_guard<std::mutex> lock(_meas_mutex);
      int c = 0;
      for (double l: vl)
	{
	  std::string measl = meas + '_' + cnames.at(c);//std::to_string(c);
	  auto hit = _meas.find(measl);
	  if (hit!=_meas.end())
	    (*hit).second = l;
	  else _meas.insert(std::pair<std::string,double>(measl,l));
	  ++c;
	}
    }
    
    /**
     * \brief get currentvalue of argument measure
     * @param meas measure name
     * @return current value of measure
     */
    double get_meas(const std::string &meas) const
    {
      std::lock_guard<std::mutex> lock(_meas_mutex);
      auto hit = _meas.find(meas);
      if (hit!=_meas.end())
	return (*hit).second;
      else return std::numeric_limits<double>::quiet_NaN();
    }

    /**
     * \brief collect current measures into a data object
     * @param ad data object to hold the measures
     */
    void collect_measures(APIData &ad) const
    {
      APIData meas;
      std::lock_guard<std::mutex> lock(_meas_mutex);
      auto hit = _meas.begin();
      while(hit!=_meas.end())
	{
	  meas.add((*hit).first,(*hit).second);
	  ++hit;
	}
      ad.add("measure",meas);
    }

    /**
     * \brief render estimated remaining time
     * @param ad data object to hold the estimate
     */
    void est_remain_time(APIData &out) const
    {
      APIData meas = out.getobj("measure");
      if (meas.has("remain_time")){    
        int est_remain_time = static_cast<int>(meas.get("remain_time").get<double>());
        int seconds = est_remain_time % 60;
        int minutes = (est_remain_time / 60) % 60;
        int hours = (est_remain_time / 60 / 60) % 24;
        int days = est_remain_time / 60 / 60 / 24;
        std::string est_remain_time_str = std::to_string(days) + "d:" + std::to_string(hours) + "h:" + std::to_string(minutes) + "m:" + std::to_string(seconds) + "s";
        meas.add("remain_time_str",est_remain_time_str);
        out.add("measure",meas);
      }
    }

    TInputConnectorStrategy _inputc; /**< input connector strategy for channeling data in. */
    TOutputConnectorStrategy _outputc; /**< output connector strategy for passing results back to API. */

    std::string _mltype = ""; /**< ml lib service instantiated type (e.g. regression, segmentation, detection, ...) */
    
    bool _has_predict = true; /**< whether prediction is available. */

    TMLModel _mlmodel; /**< statistical model template. */
    std::string _libname; /**< ml lib name. */
    
    std::unordered_map<std::string,double> _meas; /**< model measures, used as a per service value. */
    std::unordered_map<std::string,std::vector<double>> _meas_per_iter; /**< model measures per iteration. */

    std::atomic<bool> _tjob_running = {false}; /**< whether a training job is running with this lib instance. */

    bool _online = false; /**< whether the algorithm is online, i.e. it interleaves training and prediction calls.
			     When not, prediction calls are rejected while training is running. */

    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    long int _model_flops = 0;  /**< model flops. */
    long int _model_params = 0;  /**< number of parameters in the model. */
    long int _mem_used_train = 0; /**< amount  of memory used. */
    long int _mem_used_test = 0; /**< amount  of memory used. */

  protected:
    mutable std::mutex _meas_per_iter_mutex; /**< mutex over measures history. */
    mutable std::mutex _meas_mutex; /** mutex around current measures. */
    const int _max_meas_points = 1e7; // 10M points max per measure
  };  
  
}

#endif
