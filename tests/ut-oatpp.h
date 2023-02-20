/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
 * Author: Mehdi Abaakouk <mabaakouk@jolibrain.com>
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

#ifndef UTOATPP_H
#define UTOATPP_H

#include <functional>

#include "oatpp-test/UnitTest.hpp"

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/core/macro/component.hpp"
#include "oatpp/web/client/ApiClient.hpp"
#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/web/client/HttpRequestExecutor.hpp"
#include "oatpp/network/virtual_/client/ConnectionProvider.hpp"
#include "oatpp/network/virtual_/server/ConnectionProvider.hpp"
#include "oatpp/network/virtual_/Interface.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp-test/web/ClientServerTestRunner.hpp"

#include "oatppjsonapi.h"
#include "http/controller.hpp"

class TestComponent
{
public:
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::virtual_::Interface>,
                         virtualInterface)
  ([] {
    return oatpp::network::virtual_::Interface::obtainShared("virtualhost");
  }());

  OATPP_CREATE_COMPONENT(
      std::shared_ptr<oatpp::network::ServerConnectionProvider>,
      serverConnectionProvider)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::virtual_::Interface>,
                    interface);
    return oatpp::network::virtual_::server::ConnectionProvider::createShared(
        interface);
  }());

  OATPP_CREATE_COMPONENT(
      std::shared_ptr<oatpp::network::ClientConnectionProvider>,
      clientConnectionProvider)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::virtual_::Interface>,
                    interface);
    return oatpp::network::virtual_::client::ConnectionProvider::createShared(
        interface);
  }());

  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                         httpRouter)
  ([] { return oatpp::web::server::HttpRouter::createShared(); }());

  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>,
                         serverConnectionHandler)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                    router); // get Router component
    return oatpp::web::server::HttpConnectionHandler::createShared(router);
  }());

  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                         apiObjectMapper)
  ([] {
    return oatpp::parser::json::mapping::ObjectMapper::createShared();
  }());
};

#include OATPP_CODEGEN_BEGIN(ApiClient)

class DedeApiTestClient : public oatpp::web::client::ApiClient
{

  API_CLIENT_INIT(DedeApiTestClient)
  API_CALL("GET", "/info", get_info)
  API_CALL("GET", "/services/{service-name}", get_services,
           PATH(oatpp::String, service_name, "service-name"))
  API_CALL("POST", "/services/{service-name}", post_services,
           PATH(oatpp::String, service_name, "service-name"),
           BODY_STRING(oatpp::String, service_data))
  API_CALL("PUT", "/services/{service-name}", put_services,
           PATH(oatpp::String, service_name, "service-name"),
           BODY_STRING(oatpp::String, service_data))
  API_CALL("DELETE", "/services/{service-name}", delete_services,
           PATH(oatpp::String, service_name, "service-name"),
           QUERY(String, clear))
  API_CALL("POST", "/train", post_train,
           BODY_STRING(oatpp::String, train_data))
  API_CALL("GET", "/train", get_train, QUERY(String, service),
           QUERY(Int16, job), QUERY(Int16, timeout),
           QUERY(Int16, parameters_output_max_hist_points,
                 "parameters.output.max_hist_points"))
  API_CALL("DELETE", "/train", delete_train, QUERY(String, service),
           QUERY(Int16, job))
  API_CALL("POST", "/predict", post_predict,
           BODY_STRING(oatpp::String, predict_data))
};

typedef std::function<void(std::shared_ptr<DedeApiTestClient>)>
    OatppUnitTestFunc;

#include OATPP_CODEGEN_END(ApiClient)

class DedeControllerTest : public oatpp::test::UnitTest
{

public:
  OatppUnitTestFunc oatpp_unit_test_func;

  DedeControllerTest(const char *testTAG,
                     const OatppUnitTestFunc oatpp_unit_test_func)
      : UnitTest(testTAG), oatpp_unit_test_func(oatpp_unit_test_func)
  {
  }

  void onRun()
  {
    dd::OatppJsonAPI oja;
    TestComponent component;
    oatpp::test::web::ClientServerTestRunner runner;
    // using dd mapper is required to be able to serialize the dd types
    std::shared_ptr<oatpp::data::mapping::ObjectMapper> defaultObjectMapper
        = dd::oatpp_utils::createDDMapper();
    runner.addController(
        std::make_shared<DedeController>(&oja, defaultObjectMapper));
    runner.run(
        [this, &runner] {
          OATPP_COMPONENT(
              std::shared_ptr<oatpp::network::ClientConnectionProvider>,
              clientConnectionProvider);
          OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                          objectMapper);
          auto requestExecutor
              = oatpp::web::client::HttpRequestExecutor::createShared(
                  clientConnectionProvider);
          auto client
              = DedeApiTestClient::createShared(requestExecutor, objectMapper);

          this->oatpp_unit_test_func(client);
        },
        std::chrono::minutes(10) /* test timeout */);

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
};

#endif // UTOATPP_H
