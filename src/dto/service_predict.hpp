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

#ifndef DTO_SERVICE_PREDICT_H
#define DTO_SERVICE_PREDICT_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "dto/model.hpp"
#include "dto/parameters.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class ServicePredict : public oatpp::DTO
    {
      DTO_INIT(ServicePredict, DTO /* extends */)

      DTO_FIELD_INFO(service)
      {
        info->description = "Name of the service to use for prediction.";
      }
      DTO_FIELD(String, service);

      DTO_FIELD_INFO(parameters)
      {
        info->description = "Predict parameters of the service.";
      }
      DTO_FIELD(Object<Parameters>, parameters) = Parameters::createShared();
      DTO_FIELD(Vector<String>, data) = Vector<String>::createShared();

      DTO_FIELD(Boolean, has_mean_file) = false;

    public:
      /// Whether this service predict is part of a chain call or not
      bool _chain = false;

      // fields from previous chain data
      std::vector<cv::Mat> _data_raw_img;
      std::vector<std::string> _ids;
      std::vector<std::string> _meta_uris;
      std::vector<std::string> _index_uris;
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif
