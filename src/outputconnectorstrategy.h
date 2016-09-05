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

#ifndef OUTPUTCONNECTORSTRATEGY_H
#define OUTPUTCONNECTORSTRATEGY_H

#include <map>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>
#include "utils/utils.hpp"

namespace dd
{
  typedef Eigen::MatrixXd dMat;
  typedef Eigen::VectorXd dVec;
  
  /**
   * \brief bad parameter exception
   */
  class OutputConnectorBadParamException : public std::exception
  {
  public:
    OutputConnectorBadParamException(const std::string &s)
      :_s(s) {}
    ~OutputConnectorBadParamException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  /**
   * \brief internal error exception
   */
  class OutputConnectorInternalException : public std::exception
  {
  public:
    OutputConnectorInternalException(const std::string &s)
      :_s(s) {}
    ~OutputConnectorInternalException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };
  
  /**
   * \brief main output connector class
   */
  class OutputConnectorStrategy
  {
  public:
    OutputConnectorStrategy() {}
    ~OutputConnectorStrategy() {}

    /**
     * \brief output data reading
     */
    //int transform() { return 1; }
    
    /**
     * \brief initialization of output data connector
     * @param ad data object for "parameters/output"
     */
    void init(const APIData &ad);

    /**
     * \brief add prediction result to connector output
     * @param vrad vector of data objects
     */
    void add_results(const std::vector<APIData> &vrad);

    /**
     * \brief finalize output connector data
     * @param ad_in data output object from the API call
     * @param ad_out data object as the call response
     */
    void finalize(const APIData &ad_in, APIData &ad_out);
  };

  /**
   * \brief no output connector class
   */
  class NoOutputConn : public OutputConnectorStrategy
  {
  public:
    NoOutputConn()
      :OutputConnectorStrategy() {}
    ~NoOutputConn() {}
  };
  
}

#include "supervisedoutputconnector.h"
#include "unsupervisedoutputconnector.h"

#endif
