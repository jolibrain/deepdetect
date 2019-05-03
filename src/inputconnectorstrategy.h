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
#include "utils/fileops.hpp"
#ifndef WIN32
#include "utils/httpclient.hpp"
#endif
#include <spdlog/spdlog.h>
#include <exception>

namespace dd
{

  /**
   * \brief fetched data element
   * Note: functions read_mem and read_file must be defined in template type DDT
   */
  template <class DDT> class DataEl
  {
  public:
    DataEl() {}
    ~DataEl() {}

    int read_element(const std::string &uri,
		     std::shared_ptr<spdlog::logger> &logger)
    {
      _ctype._logger = logger;
      bool dir = false;
      if (uri.find("https://") != std::string::npos
	  || uri.find("http://") != std::string::npos
	  || uri.find("file://") != std::string::npos)
	{
#ifdef WIN32
      return -1;
#else
	  int outcode = -1;
	  try
	    {
	      httpclient::get_call(uri,"GET",outcode,_content);
	    }
	  catch(...)
	    {
	      throw;
	    }
	  if (outcode != 200)
	    return -1;
	  return _ctype.read_mem(_content);
#endif
	}
      else if (fileops::file_exists(uri,dir))
	{
	  if (fileops::is_db(uri))
	    return _ctype.read_db(uri); // XXX: db can acutally be a dir (e.g. lmdb)
	  else if (dir)
	    return _ctype.read_dir(uri);
	  else return _ctype.read_file(uri);
	}
      else return _ctype.read_mem(uri);
      return 0;
    }
    
    std::string _content;
    DDT _ctype;
  };
  
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
    InputConnectorStrategy(const InputConnectorStrategy &i)
      :_model_repo(i._model_repo),_logger(i._logger) {}
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

    /**
     * \brief try to acquire the input data from the main 'data' field
     *        that is mandatory for /train and /predict calls
     * @param ad root data object
     */
    void get_data(const APIData &ad)
    {
      try
	{
	  _uris = ad.get("data").get<std::vector<std::string>>();
	}
      catch(...)
	{
	  throw InputConnectorBadParamException("missing data");
	}
      if (_uris.empty())
	{
	  throw InputConnectorBadParamException("missing data");
	}
    }

    /**
     * \brief input parameters to return to user through API,
     *        especially when they have been automatically modified,
     *        and may be of use at prediction time
     * @param out output data object
     */
    void response_params(APIData &out)
    {
      (void)out;
    }
    
    bool _train = false; /**< whether in train or predict mode. */
    bool _shuffle = false; /**< whether to shuffle the dataset, usually before splitting. */

    std::vector<std::string> _uris;
    std::string _model_repo; /**< model repository, useful when connector needs to read from saved data (e.g. vocabulary). */
    std::shared_ptr<spdlog::logger> _logger;
  };
  
}

#endif
