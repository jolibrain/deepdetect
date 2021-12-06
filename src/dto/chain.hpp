/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DTO_CHAIN_HPP
#define DTO_CHAIN_HPP

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "dto/service_predict.hpp"
#include "dto/common.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    // INPUT

    class ChainActionParams : public oatpp::DTO
    {
      DTO_INIT(ChainActionParams, DTO)

      // image
      DTO_FIELD_INFO(output_images)
      {
        info->description = "If true, this action will add its result images "
                            "to the response body of the chain call";
      }
      DTO_FIELD(Boolean, output_images) = false;

      DTO_FIELD_INFO(to_rgb)
      {
        info->description = "*Removed - use `parameters.input.rgb` instead*\n"
                            "Convert image to RGB";
      }
      DTO_FIELD(Boolean, to_rgb) = false;

      DTO_FIELD_INFO(to_bgr)
      {
        info->description = "*Removed - use `parameters.input.rgb` instead*\n"
                            "Convert image to BGR";
      }
      DTO_FIELD(Boolean, to_bgr) = false;

      DTO_FIELD_INFO(save_path)
      {
        info->description = "Path to save chain output, eg for debugging";
      }
      DTO_FIELD(String, save_path) = "";

      DTO_FIELD_INFO(save_img)
      {
        info->description
            = "whether to save image to `save_path` after action";
      }
      DTO_FIELD(Boolean, save_img) = false;

      // image - crop
      DTO_FIELD_INFO(fixed_width)
      {
        info->description = "[crop] if != 0, the crop will be centered on the "
                            "center of the bbox and of width `fixed_width`";
      }
      DTO_FIELD(Int32, fixed_width) = 0;

      DTO_FIELD_INFO(fixed_height)
      {
        info->description = "[crop] if != 0, the crop will be centered on the "
                            "center of the bbox and of height `fixed_height`";
      }
      DTO_FIELD(Int32, fixed_height) = 0;

      DTO_FIELD_INFO(padding_ratio)
      {
        info->description
            = "[crop] how larger the crop should be relatively to the bbox. "
              "eg a padding_ratio of 0.1 means 10% of the size of the bbox "
              "will be added on each side of the crop";
      }
      DTO_FIELD(Float64, padding_ratio) = 0.0;

      DTO_FIELD_INFO(save_crops)
      {
        info->description = "[crop] whether to save crops to `save_path`";
      }
      DTO_FIELD(Boolean, save_crops) = false;

      // image - rotate
      DTO_FIELD_INFO(orientation)
      {
        info->description
            = "[rotate] whether rotation angle is `relative` or `absolute`";
      }
      DTO_FIELD(String, orientation) = "relative";

      // image - draw bbox
      DTO_FIELD_INFO(thickness)
      {
        info->description = "[draw_bbox] thickness of bbox rectangle";
      }
      DTO_FIELD(Int32, thickness) = 2;

      DTO_FIELD_INFO(write_cat)
      {
        info->description
            = "[draw_bbox] Write the found best class beside the bbox";
      }
      DTO_FIELD(Boolean, write_cat) = true;

      DTO_FIELD_INFO(write_prob)
      {
        info->description
            = "[draw_bbox] Write the prediction score beside the bbox";
      }
      DTO_FIELD(Boolean, write_prob) = false;

      // filter
      DTO_FIELD_INFO(classes)
      {
        info->description = "[filter] classes NOT present in this list will "
                            "be filtered out";
      }
      DTO_FIELD(Vector<String>, classes);

      // dlib image align
      DTO_FIELD(Int32, chip_size) = 150;
    };

    class ChainAction : public oatpp::DTO
    {
      DTO_INIT(ChainAction, DTO)

      DTO_FIELD_INFO(type)
      {
        info->description = "Action type, one of `crop`,`filter`,[...]";
      }
      DTO_FIELD(String, type) = "";

      DTO_FIELD(Object<ChainActionParams>, parameters)
          = ChainActionParams::createShared();
    };

    class ChainCall : public ServicePredict
    {
      DTO_INIT(ChainCall, dd::DTO::ServicePredict)

      DTO_FIELD_INFO(id)
      {
        info->description = "Chain call id, as referenced by `parent_id`";
      }
      DTO_FIELD(String, id);

      DTO_FIELD_INFO(parent_id)
      {
        info->description = "Set the input data of this calls to the "
                            "outputs of `parent_id`";
      }
      DTO_FIELD(String, parent_id);

      // action
      DTO_FIELD_INFO(action)
      {
        info->description = "Chain action in between service calls, to "
                            "perform various operation on raw/transformed "
                            "data. Mutually exclusive with `service`.";
      }
      DTO_FIELD(Object<ChainAction>, action);
    };

    class Chain : public oatpp::DTO
    {
      DTO_INIT(Chain, DTO)

      DTO_FIELD(Vector<Object<ChainCall>>, calls)
          = Vector<Object<ChainCall>>::createShared();
    };

    class ServiceChain : public oatpp::DTO
    {
      DTO_INIT(ServiceChain, DTO)

      DTO_FIELD(Object<Chain>, chain);
    };

    // OUTPUT

    class ChainHead : public oatpp::DTO
    {
      DTO_INIT(ChainHead, DTO)
    };

    // TODO rename chain output body
    class ChainBody : public oatpp::DTO
    {
      DTO_INIT(ChainBody, DTO)

      DTO_FIELD(Vector<UnorderedFields<Any>>, predictions)
          = Vector<UnorderedFields<Any>>::createShared();
      DTO_FIELD(Float64, time);
    };

    class ChainResponse : public GenericResponse
    {
      DTO_INIT(ChainResponse, GenericResponse)

      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<ChainHead>, head);
      DTO_FIELD(Object<ChainBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // DTO_CHAIN_HPP
