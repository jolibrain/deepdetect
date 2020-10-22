/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
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
#include <boost/asio.hpp>

namespace dd
{

  std::string uri_query_to_json(const std::string &req_query);

  class BeastHttpJsonAPI : public JsonAPI
  {
    boost::asio::io_context *_ioc;
    std::thread *_main_thread;

  public:
    BeastHttpJsonAPI();
    ~BeastHttpJsonAPI();

    void stop_server();
    int start_server_daemon(const std::string &host, const std::string &port,
                            const int &nthreads);
    int start_server(const std::string &host, const std::string &port,
                     const int &nthreads);
    int boot(int argc, char *argv[]);
    static void terminate(int param);

    // FIXME(sileht) should not be there ?
    void mergeJObj(JVal &to, JVal &from, JDoc &jd);
  };
}

#endif
