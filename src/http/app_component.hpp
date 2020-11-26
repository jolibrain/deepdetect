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

#ifndef HTTP_APP_HPP
#define HTTP_APP_HPP

#include "oatpp/web/protocol/http/incoming/SimpleBodyDecoder.hpp"
#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/web/server/HttpRouter.hpp"
#include "oatpp/network/ConnectionHandler.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/core/macro/component.hpp"
#include "oatpp-zlib/EncoderProvider.hpp"

#include "http/error_handler.hpp"
#include "http/swagger_component.hpp"

#include <gflags/gflags.h>

DECLARE_string(host);
DECLARE_uint32(port);

class AppComponent
{
public:
  /**
   *  Swagger component
   */
  SwaggerComponent swaggerComponent;

  /**
   * Create ObjectMapper component to serialize/deserialize DTOs in Contoller's
   * API
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                         apiObjectMapper)
  ([] {
    auto objectMapper
        = oatpp::parser::json::mapping::ObjectMapper::createShared();
    objectMapper->getDeserializer()->getConfig()->allowUnknownFields = false;
    return objectMapper;
  }());

  /**
   *  Create ConnectionProvider component which listens on the port
   */
  OATPP_CREATE_COMPONENT(
      std::shared_ptr<oatpp::network::ServerConnectionProvider>,
      serverConnectionProvider)
  ([] {
    return oatpp::network::tcp::server::ConnectionProvider::createShared({

        oatpp::base::StrBuffer::createFromCString(FLAGS_host.c_str()),
        static_cast<v_uint16>(FLAGS_port), oatpp::network::Address::IP_4 });
  }());

  /**
   *  Create Router component
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                         httpRouter)
  ([] { return oatpp::web::server::HttpRouter::createShared(); }());

  /**
   *  Create ConnectionHandler component which uses Router component to route
   * requests, and use oatpp-zlib to compress and decompress input/output
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>,
                         serverConnectionHandler)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                    router); // get Router component
    OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                    objectMapper); // get ObjectMapper component

    /* Create HttpProcessor::Components */
    auto components
        = std::make_shared<oatpp::web::server::HttpProcessor::Components>(
            router);

    /* Add content encoders */
    auto encoders = std::make_shared<
        oatpp::web::protocol::http::encoding::ProviderCollection>();

    encoders->add(std::make_shared<oatpp::zlib::DeflateEncoderProvider>());
    encoders->add(std::make_shared<oatpp::zlib::GzipEncoderProvider>());

    /* Set content encoders */
    components->contentEncodingProviders = encoders;

    /* Add content decoders */
    auto decoders = std::make_shared<
        oatpp::web::protocol::http::encoding::ProviderCollection>();

    decoders->add(std::make_shared<oatpp::zlib::DeflateDecoderProvider>());
    decoders->add(std::make_shared<oatpp::zlib::GzipDecoderProvider>());

    /* Set Body Decoder */
    components->bodyDecoder = std::make_shared<
        oatpp::web::protocol::http::incoming::SimpleBodyDecoder>(decoders);

    auto connectionHandler
        = std::make_shared<oatpp::web::server::HttpConnectionHandler>(
            components);
    connectionHandler->setErrorHandler(
        std::make_shared<ErrorHandler>(objectMapper));

    return connectionHandler;
  }());
};

#endif /* HTTP_APP_HPP */
