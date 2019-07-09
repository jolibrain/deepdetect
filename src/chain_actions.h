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
#include <iostream>

namespace dd
{

  //TODO: exception classes
  
  //TODO: chain_action class
  // - action type
  // - apply(APIData in, out) ?
  //TODO: derivatives to the chain_action class, e.g. crop_img, ...

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
    
    int apply(APIData &model_out,
	      std::unordered_map<std::string,APIData> &actions_data);

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
    
    //TODO: will except on missing data, e.g. bbox
    int apply(APIData &model_out,
	      std::unordered_map<std::string,APIData> &actions_data);
  };

  class ClassFilter : public ChainAction
  {
  public:
    ClassFilter(const APIData &adc,
		const std::string &action_type)
      :ChainAction(adc,action_type) {_in_place = true;}
    ~ClassFilter() {}

    int apply(APIData &model_out,
	      std::unordered_map<std::string,APIData> &action_out);
  };

  class ChainActionFactory
  {
  public:
    ChainActionFactory(const APIData &adc)
      :_adc(adc) {}
    ~ChainActionFactory() {}

    int apply_action(const std::string &action_type,
		     APIData &model_out,
		     std::unordered_map<std::string,APIData> &action_out)
    {
      if (action_type == "crop")
	{
	  ImgsCropAction act(_adc,action_type);
	  return act.apply(model_out,action_out);
	}
      else if (action_type == "filter")
	{
	  ClassFilter act(_adc,action_type);
	  return act.apply(model_out,action_out);
	}
      else
	{
	  //TODO: exception or ignore
	  std::cerr << "[chain] ignoring action " << action_type << std::endl;
	}
    }

    APIData _adc; /**< action ad object. */
  };
  
} // end of namespace

#endif
