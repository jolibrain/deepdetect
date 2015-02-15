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

#ifndef INPUTCONNECTORSTRATEGY_H
#define INPUTCONNECTORSTRATEGY_H

#include "apidata.h"
#include <exception>

namespace dd
{

  /**
   * \brief bad parameter exception
   */
  class InputConnectorBadParamException : public std::exception
  {
  public:
    InputConnectorBadParamException(const std::string &s)
      :_s(s) {}
    ~InputConnectorBadParamException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  /**
   * \brief internal error exception
   */
  class InputConnectorInternalException : public std::exception
  {
  public:
    InputConnectorInternalException(const std::string &s)
      :_s(s) {}
    ~InputConnectorInternalException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  /**
   * \brief main input connector class
   */
  class InputConnectorStrategy
  {
  public:
    InputConnectorStrategy() {}
    ~InputConnectorStrategy() {}
    
    /**
     * \brief initializsation of input connector
     * @param ad data object for "parameters/input"
     */
    void init(const APIData &ad);
    
    /**
     * \brief input data reading, called from ML library
     * @param ap root data object (requires access to "data" and "parameters/input")
     */
    void transform(const APIData &ap);
    
    /**
     * \brief input feature size
     */
    int feature_size() const;

    /**
     * \brief input batch size (also used for training set)
     */
    int batch_size() const;
    
    /**
     * \brief input test batch size, when applicable
     */
    int test_batch_size() const;

    bool _train = false; /**< whether in train or predict mode. */
  };
  
}

#endif
