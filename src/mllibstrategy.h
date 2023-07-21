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

#include <atomic>
#include <exception>
#include <mutex>

#include "apidata.h"
#include "service_stats.h"
#include "utils/fileops.hpp"
#include "dd_spdlog.h"
#include "dto/predict_out.hpp"

namespace dd
{
  /**
   * \brief ML library bad parameter exception
   */
  class MLLibBadParamException : public std::exception
  {
  public:
    MLLibBadParamException(const std::string &s) : _s(s)
    {
    }
    ~MLLibBadParamException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief ML library internal error exception
   */
  class MLLibInternalException : public std::exception
  {
  public:
    MLLibInternalException(const std::string &s) : _s(s)
    {
    }
    ~MLLibInternalException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief main class for machine learning library encapsulation
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  class MLLib
  {
  public:
    /**
     * \brief constructor from model
     */
    MLLib(const TMLModel &mlmodel) : _mlmodel(mlmodel), _tjob_running(false)
    {
    }

    /**
     * \brief move-constructor
     */
    MLLib(MLLib &&mll) noexcept
        : _inputc(mll._inputc), _outputc(mll._outputc), _mltype(mll._mltype),
          _mlmodel(mll._mlmodel), _meas(mll._meas),
          _meas_per_iter(mll._meas_per_iter), _stats(mll._stats),
          _tjob_running(mll._tjob_running.load()), _logger(mll._logger),
          _model_flops(mll._model_flops), _model_params(mll._model_params),
          _mem_used_train(mll._mem_used_train),
          _mem_used_test(mll._mem_used_test)
    {
    }

    /**
     * \brief destructor
     */
    ~MLLib()
    {
    }

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
        throw MLLibBadParamException("Failed opening directory "
                                     + _mlmodel._repo
                                     + " for deleting files within");
      else if (err < 0)
        throw MLLibInternalException("Failed deleting all files in directory "
                                     + _mlmodel._repo);
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
     * @param ad root input call object
     * @return result DTO containing predictions
     */
    oatpp::Object<DTO::PredictBody> predict(const APIData &ad_in);

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
      if (hit != _meas_per_iter.end())
        {
          (*hit).second.push_back(l);
          if ((int)(*hit).second.size() >= _max_meas_points)
            {
              // resolution is halved
              std::vector<double> vmeas_short;
              vmeas_short.reserve(_max_meas_points / 2);
              int di = 0;
              for (size_t j = 0; j < (*hit).second.size(); j += 2)
                vmeas_short.at(di++) = (*hit).second.at(j);
              (*hit).second = vmeas_short;
            }
        }
      else
        {
          std::vector<double> vmeas = { l };
          _meas_per_iter.insert(
              std::pair<std::string, std::vector<double>>(meas, vmeas));
        }
    }

    /**
     * \brief sub-samples measure history to fit a fixed number of points at
     * max
     * @param hist measure history vector
     * @param npoints max number of output points
     */
    int subsample_hist(const std::vector<double> &hist,
                       std::vector<double> &sub_hist, const int &npoints) const
    {
      sub_hist.clear();
      sub_hist.reserve(npoints);
      int rpoints = static_cast<int>(std::ceil(hist.size() / (double)npoints));
      for (size_t i = 0; i < hist.size(); i += rpoints)
        sub_hist.push_back(hist.at(i));
      return rpoints;
    }

    /**
     * \brief collect current measures history into a data object
     * @param ad api data object
     * @param npoints max number of output points, < 0 if unbounded
     */
    void collect_measures_history(APIData &ad, const int &npoints = -1) const
    {
      APIData meas_hist;
      APIData meas_sampling;
      std::lock_guard<std::mutex> lock(_meas_per_iter_mutex);
      auto hit = _meas_per_iter.begin();
      while (hit != _meas_per_iter.end())
        {
          if (npoints > 0 && (int)(*hit).second.size() > npoints)
            {
              std::vector<double> sub_hist;
              int sampling = subsample_hist((*hit).second, sub_hist, npoints);
              meas_hist.add((*hit).first + "_hist", sub_hist);
              meas_sampling.add((*hit).first + "_sampling", sampling);
            }
          else
            meas_hist.add((*hit).first + "_hist", (*hit).second);
          ++hit;
        }
      ad.add("measure_hist", meas_hist);
      ad.add("measure_sampling", meas_sampling);
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
      for (auto s : mkeys)
        {
          std::vector<double> mdata
              = ad_metrics.get(s).get<std::vector<double>>();
          s.replace(s.find("_hist"), 5, "");
          _meas_per_iter.insert(
              std::pair<std::string, std::vector<double>>(s, mdata));
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
      if (hit != _meas.end())
        (*hit).second = l;
      else
        _meas.insert(std::pair<std::string, double>(meas, l));
    }

    void add_meas(const std::string &meas, const std::vector<double> &vl,
                  const std::vector<std::string> &cnames)
    {
      std::lock_guard<std::mutex> lock(_meas_mutex);
      int c = 0;
      for (double l : vl)
        {
          std::string measl = meas + '_' + cnames.at(c);
          auto hit = _meas.find(
              measl); // not reusing add_meas since need a global lock
          if (hit != _meas.end())
            (*hit).second = l;
          else
            _meas.insert(std::pair<std::string, double>(measl, l));
          ++c;
        }
    }

    /**
     * \brief get current value of argument measure
     * @param meas measure name
     * @return current value of measure
     */
    double get_meas(const std::string &meas) const
    {
      std::lock_guard<std::mutex> lock(_meas_mutex);
      auto hit = _meas.find(meas);
      if (hit != _meas.end())
        return (*hit).second;
      else
        return std::numeric_limits<double>::quiet_NaN();
    }

    /**
     * \brief collect current measures into a data object
     * @param ad data object to hold the measures
     */
    void collect_measures(APIData &ad) const
    {
      APIData meas;

      {
        std::lock_guard<std::mutex> lock(_meas_mutex);
        auto hit = _meas.begin();
        while (hit != _meas.end())
          {
            meas.add((*hit).first, (*hit).second);
            ++hit;
          }
      }

      if (meas.has("remain_time"))
        {
          int est_remain_time
              = static_cast<int>(meas.get("remain_time").get<double>());
          int seconds = est_remain_time % 60;
          int minutes = (est_remain_time / 60) % 60;
          int hours = (est_remain_time / 60 / 60) % 24;
          int days = est_remain_time / 60 / 60 / 24;
          std::string est_remain_time_str
              = std::to_string(days) + "d:" + std::to_string(hours)
                + "h:" + std::to_string(minutes)
                + "m:" + std::to_string(seconds) + "s";
          meas.add("remain_time_str", est_remain_time_str);
        }

      meas.add("flops", this->_model_flops);

      APIData test_names;
      for (size_t i = 0; i < _test_names.size(); ++i)
        {
          test_names.add(std::to_string(i), _test_names[i]);
        }
      meas.add("test_names", test_names);

      ad.add("measure", meas);
    }

    TInputConnectorStrategy
        _inputc; /**< input connector strategy for channeling data in. */
    TOutputConnectorStrategy _outputc; /**< output connector strategy for
                                          passing results back to API. */

    std::string _mltype = ""; /**< ml lib service instantiated type (e.g.
                                 regression, segmentation, detection, ...) */

    bool _has_predict = true; /**< whether prediction is available. */

    TMLModel _mlmodel;    /**< statistical model template. */
    std::string _libname; /**< ml lib name. */

    std::unordered_map<std::string, double>
        _meas; /**< model measures, used as a per service value. */
    std::unordered_map<std::string, std::vector<double>>
        _meas_per_iter; /**< model measures per iteration. */
    std::vector<std::string> _test_names;

    ServiceStats _stats; /**< service statistics/metrics .*/

    std::atomic<bool> _tjob_running = {
      false
    }; /**< whether a training job is running with this lib instance. */

    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    long int _model_flops = 0;  /**< model flops. */
    long int _model_params = 0; /**< number of parameters in the model. */
    long int _model_frozen_params
        = 0; /**< number of frozen parameters in the model. */
    long int _mem_used_train = 0; /**< amount  of memory used. */
    long int _mem_used_test = 0;  /**< amount  of memory used. */

  protected:
    mutable std::mutex
        _meas_per_iter_mutex;         /**< mutex over measures history. */
    mutable std::mutex _meas_mutex;   /**< mutex around current measures. */
    const int _max_meas_points = 1e7; // 10M points max per measure
  };

}

#endif
