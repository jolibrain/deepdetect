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

namespace dd
{
  class OatppJsonAPI : public JsonAPI
  {
    std::shared_ptr<spdlog::logger> _logger;

  public:
    OatppJsonAPI();
    ~OatppJsonAPI();

    int boot(int argc, char *argv[]);
    void run();
    static void terminate(int signal);
    static void abort(int param);
    std::string
    uri_query_to_json(oatpp::web::protocol::http::QueryParams queryParams);
    std::shared_ptr<oatpp::web::protocol::http::outgoing::Response>
    jdoc_to_response(const JDoc &janswer);
  };
}

#endif
