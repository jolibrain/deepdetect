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
      :_mlmodel(mlmodel),_loss(0.0),_tjob_running(false) {}
    
    /**
     * \brief copy-constructor
     */
    MLLib(MLLib &&mll) noexcept
      :_inputc(mll._inputc),_outputc(mll._outputc),_mlmodel(mll._mlmodel),_loss(mll._loss.load()),_tjob_running(mll._tjob_running.load())
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
     * \brief clear loss history
     */
    void clear_loss_per_iter()
    {
      std::lock_guard<std::mutex> lock(_loss_per_iter_mutex);
      _loss_per_iter.clear();
    }

    /**
     * \brief add value to loss history
     * @param l loss value
     */
    void add_loss_per_iter(const double &l)
    {
      std::lock_guard<std::mutex> lock(_loss_per_iter_mutex);
      _loss_per_iter.push_back(l);
    }

    TInputConnectorStrategy _inputc; /**< input connector strategy for channeling data in. */
    TOutputConnectorStrategy _outputc; /**< output connector strategy for passing results back to API. */

    bool _has_train = false; /**< whether training is available. */
    bool _has_predict = true; /**< whether prediction is available. */

    TMLModel _mlmodel; /**< statistical model template. */
    std::string _libname; /**< ml lib name. */
    
    std::atomic<double> _loss = 0.0; /**< model loss, used as a per service value. */
    std::vector<double> _loss_per_iter; /**< model loss per iteration. */

    std::atomic<bool> _tjob_running = false; /**< whether a training job is running with this lib instance. */

  protected:
    std::mutex _loss_per_iter_mutex; /**< mutex over loss history. */
  };  
  
}

#endif
