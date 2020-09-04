/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef HTTPJSONAPI_H
#define HTTPJSONAPI_H

#include "jsonapi.h"
#include <boost/network/protocol/http/server.hpp>
#include <boost/network/uri.hpp>
#include <boost/network/uri/uri_io.hpp>

namespace http = boost::network::http;
namespace uri = boost::network::uri;
class APIHandler;
typedef http::server<APIHandler> http_server;

namespace dd
{
  std::string uri_query_to_json(const std::string &req_query);

  class HttpJsonAPI : public JsonAPI
  {
  public:
    HttpJsonAPI();
    ~HttpJsonAPI();

    void stop_server();
    int start_server_daemon(const std::string &host, const std::string &port,
                            const int &nthreads);
    int start_server(const std::string &host, const std::string &port,
                     const int &nthreads);
    int boot(int argc, char *argv[]);
    static void terminate(int param);
    void mergeJObj(JVal &to, JVal &from, JDoc &jd);

    http_server *_dd_server
        = nullptr;        /**< main reusable pointer to server object */
    std::future<int> _ft; /**< holds the results from the main server thread */
  };
}

#endif
