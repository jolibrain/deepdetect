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

#ifndef CRUD_STATICCONTROLLER_HPP
#define CRUD_STATICCONTROLLER_HPP

#include <memory>
#include <vector>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/macro/component.hpp"

#include "apidata.h"
#include "oatppjsonapi.h"
#include "dto/info.hpp"

#include OATPP_CODEGEN_BEGIN(ApiController)

class DedeController : public oatpp::web::server::api::ApiController
{
public:
  DedeController(dd::OatppJsonAPI *oja,
                 const std::shared_ptr<ObjectMapper> &objectMapper)
      : oatpp::web::server::api::ApiController(objectMapper), _oja(oja)
  {
  }

private:
  dd::OatppJsonAPI *_oja = nullptr;

public:
  static std::shared_ptr<DedeController>
  createShared(dd::OatppJsonAPI *oja = nullptr,
               OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper))
  {
    return std::make_shared<DedeController>(oja, objectMapper);
  }

  ENDPOINT_INFO(get_info)
  {
    info->summary = "Retreive server information";
  }
  ENDPOINT("GET", "info", get_info, QUERIES(QueryParams, queryParams))
  {
    auto info_resp = dd::DTO::InfoResponse::createShared();
    info_resp->head = dd::DTO::InfoHead::createShared();
    info_resp->head->services = {};

    auto qs_status = queryParams.get("status");
    bool status = false;
    if (qs_status)
      status = boost::lexical_cast<bool>(qs_status->std_str());

    auto hit = _oja->_mlservices.begin();
    while (hit != _oja->_mlservices.end())
      {
        // TODO(sileht): update visitor_info to return directly a Service()
        JDoc jd;
        jd.SetObject();
        mapbox::util::apply_visitor(dd::visitor_info(status), (*hit).second)
            .toJDoc(jd);
        auto json_str = _oja->jrender(jd);
        auto service_info
            = getDefaultObjectMapper()
                  ->readFromString<oatpp::Object<dd::DTO::Service>>(
                      json_str.c_str());
        info_resp->head->services->emplace_back(service_info);
        ++hit;
      }
    return createDtoResponse(Status::CODE_200, info_resp);
  }

  ENDPOINT_INFO(get_service)
  {
    info->summary = "Retreive a service detail";
  }
  ENDPOINT("GET", "services/{service-name}", get_service,
           PATH(oatpp::String, service_name, "service-name"))
  {
    auto janswer = _oja->service_status(service_name.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(create_service)
  {
    info->summary = "Create a service";
  }
  ENDPOINT("POST", "services/{service-name}", create_service,
           PATH(oatpp::String, service_name, "service-name"),
           BODY_STRING(oatpp::String, service_data))
  {
    auto janswer = _oja->service_create(service_name.get()->std_str(),
                                        service_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(update_service)
  {
    // Don't document PUT, it's a dup of POST, maybe deprecate it later
    info->hide = true;
  }
  ENDPOINT("PUT", "services/{service-name}", update_service,
           PATH(oatpp::String, service_name, "service-name"),
           BODY_STRING(oatpp::String, service_data))
  {
    auto janswer = _oja->service_create(service_name.get()->std_str(),
                                        service_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }
  ENDPOINT_INFO(delete_service)
  {
    info->summary = "Delete a service";
  }
  ENDPOINT("DELETE", "services/{service-name}", delete_service,
           PATH(oatpp::String, service_name, "service-name"),
           QUERIES(QueryParams, queryParams))
  {
    std::string jsonstr = _oja->uri_query_to_json(queryParams);
    auto janswer
        = _oja->service_delete(service_name.get()->std_str(), jsonstr);
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(predict)
  {
    info->summary = "Predict";
  }
  ENDPOINT("POST", "predict", predict,
           BODY_STRING(oatpp::String, predict_data))
  {
    auto janswer = _oja->service_predict(predict_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(get_train)
  {
    info->summary = "Retreive a training status";
  }
  ENDPOINT("GET", "train", get_train, QUERIES(QueryParams, queryParams))
  {
    std::string jsonstr = _oja->uri_query_to_json(queryParams);
    auto janswer = _oja->service_train_status(jsonstr);
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(post_train)
  {
    info->summary = "Do a training";
  }
  ENDPOINT("POST", "train", post_train, BODY_STRING(oatpp::String, train_data))
  {
    auto janswer = _oja->service_train(train_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(put_train)
  {
    // Don't document PUT, it's a dup of POST, maybe deprecate it later
    info->hide = true;
  }
  ENDPOINT("PUT", "train", put_train,

           BODY_STRING(oatpp::String, train_data))
  {
    auto janswer = _oja->service_train(train_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }
  ENDPOINT_INFO(delete_train)
  {
    info->summary = "Delete a training";
  }
  ENDPOINT("DELETE", "train", delete_train, QUERIES(QueryParams, queryParams))
  {
    std::string jsonstr = _oja->uri_query_to_json(queryParams);
    auto janswer = _oja->service_train_delete(jsonstr);
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(create_chain)
  {
    info->summary = "Run a chain";
  }
  ENDPOINT("POST", "chain/{chain-name}", create_chain,
           PATH(oatpp::String, chain_name, "chain-name"),
           BODY_STRING(oatpp::String, chain_data))
  {
    auto janswer = _oja->service_chain(chain_name.get()->std_str(),
                                       chain_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }

  ENDPOINT_INFO(update_chain)
  {
    // Don't document PUT, it's a dup of POST, maybe deprecate it later
    info->hide = true;
  }
  ENDPOINT("PUT", "chain/{chain-name}", update_chain,
           PATH(oatpp::String, chain_name, "chain-name"),
           BODY_STRING(oatpp::String, chain_data))
  {
    auto janswer = _oja->service_chain(chain_name.get()->std_str(),
                                       chain_data.get()->std_str());
    return _oja->jdoc_to_response(janswer);
  }
};

#include OATPP_CODEGEN_END(ApiController)

#endif // CRUD_STATICCONTROLLER_HPP
