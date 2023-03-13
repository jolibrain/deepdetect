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

#ifndef HTTPAOTJSONAPI_H
#define HTTPAOTJSONAPI_H

#include "jsonapi.h"
#include "oatpp/network/Server.hpp"
#include "oatpp/web/protocol/http/Http.hpp"
#include "oatpp/web/protocol/http/outgoing/Response.hpp"
#include "oatpp/web/server/api/ApiController.hpp"
#include "dto/common.hpp"

namespace dd
{
  class OatppJsonAPI : public JsonAPI
  {
  public:
    OatppJsonAPI();
    ~OatppJsonAPI();

    typedef oatpp::web::protocol::http::outgoing::Response Response;
    typedef std::shared_ptr<Response> Response_ptr;

    int boot(int argc, char *argv[]);
    void run();
    static void terminate(int signal);
    static void abort(int param);
    std::string
    uri_query_to_json(oatpp::web::protocol::http::QueryParams queryParams);
    Response_ptr jdoc_to_response(const JDoc &janswer) const;

    oatpp::Object<DTO::Status>
    create_status_dto(const uint32_t &code, const std::string &msg,
                      const uint32_t &dd_code = 0,
                      const std::string &dd_msg = "") const;

    Response_ptr dto_to_response(oatpp::Void dto, const uint32_t &code,
                                 const std::string &msg,
                                 const uint32_t &dd_code = 0,
                                 const std::string &dd_msg = "") const;

    void addPrefixToEndpoints(oatpp::web::server::api::Endpoints &endpoints,
                              const std::string &prefix);

    // Oatpp responses
    Response_ptr response_bad_request_400(const std::string &msg = "") const;
    Response_ptr response_not_found_404() const;
    Response_ptr response_internal_error_500(const std::string &msg
                                             = "") const;

    // dede error responses
    Response_ptr response_resource_already_exists_1015() const;
  };
}

#endif
