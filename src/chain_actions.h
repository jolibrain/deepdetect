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
#include "chain.h"

#ifdef USE_DLIB
#include "opencv2/opencv.hpp"
#include "dlib/data_io.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/image_processing.h"
#include "dlib/opencv/to_open_cv.h"
#include "dlib/opencv/cv_image.h"
#endif

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
	      const std::string &action_id,
	      const std::string &action_type)
    :_action_id(action_id),_action_type(action_type)
    {
      APIData action_adc = adc.getobj("action");
      _params = action_adc.getobj("parameters");
    }

    ~ChainAction() {}

    std::string genid(const std::string &uri,
		      const std::string &local_id)
      {
	std::string str = uri+local_id;
	return std::to_string(std::hash<std::string>{}(str));
      }
    
    void apply(APIData &model_out,
	       ChainData &cdata);

    std::string _action_id;
    std::string _action_type;
    APIData _params;
    bool _in_place = false;
  };

#ifdef USE_DLIB
  class DlibShapePredictorAction : public ChainAction
  {
  public:
    DlibShapePredictorAction(const APIData &adc,
		   const std::string &action_id,
		   const std::string &action_type)
      :ChainAction(adc,action_id,action_type) {
        dlib::deserialize(_shape_predictor_path) >> _shapePredictor;
    }

    ~DlibShapePredictorAction() {}

    void apply(APIData &model_out,
	       ChainData &cdata);

    std::string _shape_predictor_path = "shape_predictor_5_face_landmarks.dat";
    dlib::shape_predictor _shapePredictor;
  };


#endif

  class ImgsCropAction : public ChainAction
  {
  public:
    ImgsCropAction(const APIData &adc,
		   const std::string &action_id,
		   const std::string &action_type)
      :ChainAction(adc,action_id,action_type) {}

    ~ImgsCropAction() {}
    
    void apply(APIData &model_out,
	       ChainData &cdata);
  };

  class ClassFilter : public ChainAction
  {
  public:
    ClassFilter(const APIData &adc,
		const std::string &action_id,
		const std::string &action_type)
      :ChainAction(adc,action_id,action_type) {_in_place = true;}
    ~ClassFilter() {}

    void apply(APIData &model_out,
	       ChainData &cdata);
  };

  class ChainActionFactory
  {
  public:
    ChainActionFactory(const APIData &adc)
      :_adc(adc) {}
    ~ChainActionFactory() {}

    void apply_action(const std::string &action_type,
		      APIData &model_out,
		      ChainData &cdata)
    {
      std::string action_id;
      if (_adc.has("id"))
	action_id = _adc.get("id").get<std::string>();
      else action_id = std::to_string(cdata._action_data.size());
      if (action_type == "crop")
	{
	  ImgsCropAction act(_adc,action_id,action_type);
	  act.apply(model_out,cdata);
	}
      else if (action_type == "filter")
	{
	  ClassFilter act(_adc,action_id,action_type);
	  act.apply(model_out,cdata);
	}
#ifdef USE_DLIB
      else if (action_type == "dlib_shape_predictor")
    {
        DlibShapePredictorAction act(_adc,action_id,action_type);
        act.apply(model_out,cdata);
    }
#endif
      else
	{
	  throw ActionBadParamException("unknown action " + action_type);
	}
    }

    APIData _adc; /**< action ad object. */
  };
  
} // end of namespace

#endif
