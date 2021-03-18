/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Mehdi Abaakouk <mehdi.abaakouk@jolibrain.com>
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

#ifndef HTTP_DTO_PREDICT_H
#define HTTP_DTO_PREDICT_H
#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

class PredictOutputParametersDto : public oatpp::DTO
{
  DTO_INIT(PredictOutputParametersDto, DTO /* extends */)

  /* ncnn */
  DTO_FIELD(Boolean, bbox) = false;
  DTO_FIELD(Boolean, ctc) = false;
  DTO_FIELD(Int32, blank_label) = -1;
  DTO_FIELD(Float32, confidence_threshold) = 0.0;

  /* ncnn && supervised init && supervised predict */
  DTO_FIELD(Int32, best) = -1;

  /* output supervised init */
  DTO_FIELD(Boolean, nclasses) = false; // Looks like a bug ?

  /* output supervised predict */
  DTO_FIELD(Boolean, index) = false;
  DTO_FIELD(Boolean, build_index) = false;
  DTO_FIELD(Boolean, search) = false;
  DTO_FIELD(Int32, search_nn);
  DTO_FIELD(Int32, nprobe);
};

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section

#endif
