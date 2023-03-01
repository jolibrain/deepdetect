/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
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

#ifndef HTTP_DTO_INFO_H
#define HTTP_DTO_INFO_H

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "common.hpp"
#include "ddtypes.hpp"
#include "parameters.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class ServiceModel : public oatpp::DTO
    {
      DTO_INIT(ServiceModel, DTO /* extends */)

      DTO_FIELD_INFO(flops)
      {
        info->description = "Number of flops of the model";
      }
      DTO_FIELD(Int32, flops);

      DTO_FIELD_INFO(params)
      {
        info->description = "Number of parameters of the model";
      }
      DTO_FIELD(Int32, params);

      DTO_FIELD_INFO(frozen_params)
      {
        info->description = "Number of frozen parameters in the model";
      }
      DTO_FIELD(Int32, frozen_params);

      DTO_FIELD_INFO(data_mem_train)
      {
        info->description = "Amount of memory used to store train data";
      }
      DTO_FIELD(Int32, data_mem_train);

      DTO_FIELD_INFO(data_mem_test)
      {
        info->description = "Amount of memory used to store test data";
      }
      DTO_FIELD(Int32, data_mem_test);
    };

    class ServiceJob : public oatpp::DTO
    {
      DTO_INIT(ServiceJob, DTO /* extends */)

      DTO_FIELD_INFO(job)
      {
        info->description = "Id of the job";
      }
      DTO_FIELD(Int32, job);

      DTO_FIELD_INFO(status)
      {
        info->description = "status of the job, one of: \"not started\", "
                            "\"running\", \"finished\"";
      }
      DTO_FIELD(String, status);
    };

    class Service : public oatpp::DTO
    {
      DTO_INIT(Service, DTO /* extends */)

      DTO_FIELD(String, name);
      DTO_FIELD(String, description);
      DTO_FIELD(String, mllib);
      DTO_FIELD(String, mltype);
      DTO_FIELD(Object<Parameters>, parameters);

      DTO_FIELD_INFO(type)
      {
        info->description = "supervised, unsupervised";
      }
      DTO_FIELD(String, type);

      DTO_FIELD(Boolean, predict) = false;
      DTO_FIELD(Boolean, training) = false;

      DTO_FIELD_INFO(stats)
      {
        info->description = "[deprecated] replaced by model_stats";
      }
      DTO_FIELD(Object<ServiceModel>, stats);
      DTO_FIELD(Object<ServiceModel>, model_stats);
      DTO_FIELD(Vector<DTOApiData>, jobs);

      DTO_FIELD_INFO(labels)
      {
        info->description
            = "Labels for classification / detection / segmentation services";
      }
      DTO_FIELD(Vector<String>, labels);

      DTO_FIELD(String, repository);
      DTO_FIELD(Int32, width);
      DTO_FIELD(Int32, height);

      DTO_FIELD(DTOApiData, service_stats);
    };

    class InfoHead : public oatpp::DTO
    {
      DTO_INIT(InfoHead, DTO /* extends */)
      DTO_FIELD(String, method) = "/info";

      // Why this is not in body ?
      DTO_FIELD(String, build_type, "build-type") = BUILD_TYPE;
      DTO_FIELD(String, version) = GIT_VERSION;
      DTO_FIELD(String, branch) = GIT_BRANCH;
      DTO_FIELD(String, commit) = GIT_COMMIT_HASH;
      DTO_FIELD(String, compile_flags) = COMPILE_FLAGS;
      DTO_FIELD(String, deps_version) = DEPS_VERSION;
      DTO_FIELD(List<Object<Service>>, services);
    };

    class InfoBody : public oatpp::DTO
    {
      DTO_INIT(InfoBody, DTO /* extends */)
    };

    class InfoResponse : public oatpp::DTO
    {
      DTO_INIT(InfoResponse, DTO /* extends */)
      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<InfoHead>, head);
      DTO_FIELD(Object<InfoBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // HTTP_DTO_INFO_H
