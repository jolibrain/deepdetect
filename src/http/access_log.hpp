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

#ifndef HTTP_ACCESSLOG_INTERCEPTOR_HPP
#define HTTP_ACCESSLOG_INTERCEPTOR_HPP

#include "oatpp/web/protocol/http/outgoing/Response.hpp"
#include "oatpp/web/protocol/http/incoming/Request.hpp"
#include "oatpp/web/server/interceptor/ResponseInterceptor.hpp"
#include "oatpp/web/server/interceptor/RequestInterceptor.hpp"

#include "dd_spdlog.h"

namespace dd
{
  namespace http
  {
    /* For storing per request information, mainly to build a proper access
       log */
    class AccessLogContext
    {
    public:
      std::string service_name = "<n/a>";
      std::chrono::time_point<std::chrono::steady_clock> req_start_time;
    };

    thread_local extern AccessLogContext _context;

    inline void initAccessLogRequestStartTime()
    {
      _context.req_start_time = std::chrono::steady_clock::now();
    }
    inline void setAccessLogServiceName(std::string service_name)
    {
      _context.service_name = service_name;
    }

    class AccessLogResponseInterceptor
        : public oatpp::web::server::interceptor::ResponseInterceptor
    {
    private:
      std::shared_ptr<spdlog::logger> _logger;

    public:
      AccessLogResponseInterceptor(
          const std::shared_ptr<spdlog::logger> &logger)
          : oatpp::web::server::interceptor::ResponseInterceptor(),
            _logger(logger)
      {
      }

      std::shared_ptr<OutgoingResponse>
      intercept(const std::shared_ptr<IncomingRequest> &request,
                const std::shared_ptr<OutgoingResponse> &response) override
      {

        auto req = request->getStartingLine();
        std::string access_log = req.protocol.toString() + " \""
                                 + req.method.toString() + " "
                                 + req.path.toString() + "\"";
        access_log += " " + _context.service_name;

        auto outcode = response->getStatus().code;

        access_log += " " + std::to_string(outcode);

        auto req_stop_time = std::chrono::steady_clock::now();
        auto req_duration_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                req_stop_time - _context.req_start_time);
        access_log += " " + std::to_string(req_duration_ms.count()) + "ms";

        if (outcode == 200 || outcode == 201)
          _logger->info(access_log);
        else
          _logger->error(access_log);

        return response;
      }
    };

    class AccessLogRequestInterceptor
        : public oatpp::web::server::interceptor::RequestInterceptor
    {
    public:
      std::shared_ptr<OutgoingResponse>
      intercept(const std::shared_ptr<IncomingRequest> &request) override
      {
        (void)request;
        initAccessLogRequestStartTime();
        return nullptr;
      }
    };
  }
}
#endif
