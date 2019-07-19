/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#ifndef CHAIN_ACTIONS_H
#define CHAIN_ACTIONS_H

#include "apidata.h"

namespace dd
{

  class ActionBadParamException : public std::exception
  {
  public:
    ActionBadParamException(const std::string &s)
      :_s(s) {}
    ~ActionBadParamException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };

  class ActionInternalException : public std::exception
  {
  public:
    ActionInternalException(const std::string &s)
      :_s(s) {}
    ~ActionInternalException() {}
    const char* what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };
  
  class ChainAction
  {
  public:
  ChainAction(const APIData &adc,
	      const std::string &action_type)
    :_action_type(action_type)
    {
      _params = adc.getobj("parameters");
    }

    ~ChainAction() {}

    std::string genid(const std::string &uri,
		      const std::string &local_id)
      {
	std::string str = uri+local_id;
	return std::to_string(std::hash<std::string>{}(str));
      }
    
    void apply(APIData &model_out,
	       std::vector<APIData> &actions_data);

    std::string _action_type;
    APIData _params;
    bool _in_place = false;
  };


  class ImgsCropAction : public ChainAction
  {
  public:
    ImgsCropAction(const APIData &adc,
		   const std::string &action_type)
      :ChainAction(adc,action_type) {}

    ~ImgsCropAction() {}
    
    void apply(APIData &model_out,
	       std::vector<APIData> &actions_data);
  };

  class ClassFilter : public ChainAction
  {
  public:
    ClassFilter(const APIData &adc,
		const std::string &action_type)
      :ChainAction(adc,action_type) {_in_place = true;}
    ~ClassFilter() {}

    void apply(APIData &model_out,
	       std::vector<APIData> &action_data);
  };

  class ChainActionFactory
  {
  public:
    ChainActionFactory(const APIData &adc)
      :_adc(adc) {}
    ~ChainActionFactory() {}

    void apply_action(const std::string &action_type,
		      APIData &model_out,
		      std::vector<APIData> &action_out)
    {
      if (action_type == "crop")
	{
	  ImgsCropAction act(_adc,action_type);
	  act.apply(model_out,action_out);
	}
      else if (action_type == "filter")
	{
	  ClassFilter act(_adc,action_type);
	  act.apply(model_out,action_out);
	}
      else
	{
	  throw ActionBadParamException("unknown action " + action_type);
	}
    }

    APIData _adc; /**< action ad object. */
  };
  
} // end of namespace

#endif
