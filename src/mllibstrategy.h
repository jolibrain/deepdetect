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

#ifndef MLLIBSTRATEGY_H
#define MLLIBSTRATEGY_H

#include "apidata.h"
#include <atomic>
#include <exception>
#include <mutex>

namespace dd
{
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

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    class MLLib
  {
  public:
    MLLib(const TMLModel &mlmodel)
      :_mlmodel(mlmodel),_loss(0.0),_tjob_running(false) {}
    
    MLLib(MLLib &&mll) noexcept
      :_mlmodel(mll._mlmodel),_inputc(mll._inputc),_loss(mll._loss.load()),_tjob_running(mll._tjob_running.load())
      {}
    
    ~MLLib() {}

    int train(const APIData &ad, APIData &out);
    int predict(const APIData &ad, APIData &out);
    int status() const;
    
    void clear_loss_per_iter()
    {
      std::lock_guard<std::mutex> lock(_loss_per_iter_mutex);
      _loss_per_iter.clear();
    }

    void add_loss_per_iter(const double &l)
    {
      std::lock_guard<std::mutex> lock(_loss_per_iter_mutex);
      _loss_per_iter.push_back(l);
    }

    TInputConnectorStrategy _inputc;
    TOutputConnectorStrategy _outputc;

    bool _has_train = false; /**< whether training is available. */
    bool _has_predict = true; /**< whether prediction is available. */

    TMLModel _mlmodel;
    std::string _libname; /**< ml lib name. */
    
    std::atomic<double> _loss = 0.0; /**< model loss, used as a per service value. */
    std::vector<double> _loss_per_iter; /**< model loss per iteration. */
    std::mutex _loss_per_iter_mutex;
    std::atomic<bool> _tjob_running = false; /**< whether a training job is running with this lib instance. */
  };  
  
}

#endif
