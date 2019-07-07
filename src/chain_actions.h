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
    ChainAction(const std::string &action_type)
      :_action_type(action_type) {}

    ~ChainAction() {}

    int apply(APIData &model_out,
	      std::unordered_map<std::string,APIData> &actions_data);

    std::string _action_type;
    bool _in_place = false;
  };


  class ImgsCropAction : public ChainAction
  {
  public:
    ImgsCropAction(const std::string &action_type)
      :ChainAction(action_type) {}

    ~ImgsCropAction() {}

    std::string genid(const int &i)
      {
	return "bbox_" + std::to_string(i); //TODO: needs UUID (from URI + bbox number) if batch_size > 1
      }
    
    //TODO: will except on missing data, e.g. bbox
    int apply(APIData &model_out,
	      std::unordered_map<std::string,APIData> &actions_data);
  };

  class ClassFilter : public ChainAction
  {
  public:
    ClassFilter(const std::string &action_type)
      :ChainAction(action_type) {_in_place = true;}
    ~ClassFilter() {}
  };

  class ChainActionFactory
  {
  public:
    ChainActionFactory() {}
    ~ChainActionFactory() {}

    int apply_action(const std::string &action_type,
		     APIData &model_out,
		     std::unordered_map<std::string,APIData> &action_out)
    {
      if (action_type == "crop")
	{
	  ImgsCropAction act(action_type);
	  return act.apply(model_out,action_out);
	}
      else
	{
	  //TODO: exception or ignore
	  std::cerr << "[chain] ignoring action " << action_type << std::endl;
	}
    }
  };
  
} // end of namespace

#endif
