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

#include <csignal>
#if USE_BOOST_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

#include "oatppjsonapi.h"
#include "http/app_component.hpp"
#include "http/controller.hpp"

#include "oatpp/network/Server.hpp"
#include "oatpp/web/protocol/http/Http.hpp"
#include "oatpp/web/protocol/http/outgoing/ResponseFactory.hpp"
#include "oatpp/web/server/HttpConnectionHandler.hpp"
#include "oatpp/web/server/HttpRouter.hpp"
#include "oatpp/web/server/api/ApiController.hpp"
#include "oatpp/network/tcp/server/ConnectionProvider.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/core/macro/component.hpp"
#include "oatpp-swagger/Controller.hpp"

#include <gflags/gflags.h>

DECLARE_string(allow_origin);

namespace dd
{
  oatpp::network::Server *_server = nullptr;

  OatppJsonAPI::OatppJsonAPI() : JsonAPI()
  {
  }

  OatppJsonAPI::~OatppJsonAPI()
  {
  }

  static void mergeJObj(JVal &to, JVal &from, JDoc &jd)
  {
    for (auto fromIt = from.MemberBegin(); fromIt != from.MemberEnd();
         ++fromIt)
      {
        auto toIt = to.FindMember(fromIt->name);
        if (toIt == to.MemberEnd())
          to.AddMember(fromIt->name, fromIt->value, jd.GetAllocator());
        else
          {
            if (fromIt->value.IsArray())
              for (auto arrayIt = fromIt->value.Begin();
                   arrayIt != fromIt->value.End(); ++arrayIt)
                toIt->value.PushBack(*arrayIt, jd.GetAllocator());
            else if (fromIt->value.IsObject())
              mergeJObj(toIt->value, fromIt->value, jd);
            else
              toIt->value = fromIt->value;
          }
      }
  }

  std::string OatppJsonAPI::uri_query_to_json(
      oatpp::web::protocol::http::QueryParams queryParams)
  {
    // mimic uri_query_to_json of HttpJsonApi
    JDoc jd;
    JVal jsv(rapidjson::kObjectType);
    jd.SetObject();

    for (auto &param : queryParams.getAll())
      {
        std::string qs_key = param.first.std_str();
        std::string qs_value = param.second.std_str();

        bool is_word = false;
        for (size_t i = 0; i < qs_value.size(); i++)
          {
            if (isalpha(qs_value[i]))
              {
                is_word = true;
                break;
              }
          }

        // we break '.' into JSON sub-objects
        std::vector<std::string> vpt = dd::dd_utils::split(qs_key, '.');
        JVal jobj(rapidjson::kObjectType);
        if (vpt.size() > 1)
          {
            bool bt = dd::dd_utils::iequals(qs_value, "true");
            bool bf = dd::dd_utils::iequals(qs_value, "false");
            if (is_word && !bt && !bf)
              {
                jobj.AddMember(
                    JVal().SetString(vpt.back().c_str(), jd.GetAllocator()),
                    JVal().SetString(qs_value.c_str(), jd.GetAllocator()),
                    jd.GetAllocator());
              }
            else if (bt || bf)
              {
                jobj.AddMember(
                    JVal().SetString(vpt.back().c_str(), jd.GetAllocator()),
                    JVal(bt ? true : false), jd.GetAllocator());
              }
            else
              jobj.AddMember(
                  JVal().SetString(vpt.back().c_str(), jd.GetAllocator()),
                  JVal(atoi(qs_value.c_str())), jd.GetAllocator());
            for (int b = vpt.size() - 2; b > 0; b--)
              {
                JVal jnobj(rapidjson::kObjectType);
                jobj = jnobj.AddMember(
                    JVal().SetString(vpt.at(b).c_str(), jd.GetAllocator()),
                    jobj, jd.GetAllocator());
              }
            JVal jsv2(rapidjson::kObjectType);
            jsv2.AddMember(
                JVal().SetString(vpt.at(0).c_str(), jd.GetAllocator()), jobj,
                jd.GetAllocator());
            mergeJObj(jsv, jsv2, jd);
          }
        else
          {
            bool bt = dd::dd_utils::iequals(qs_value, "true");
            bool bf = dd::dd_utils::iequals(qs_value, "false");
            if (is_word && !bt && !bf)
              {
                jsv.AddMember(
                    JVal().SetString(qs_key.c_str(), jd.GetAllocator()),
                    JVal().SetString(qs_value.c_str(), jd.GetAllocator()),
                    jd.GetAllocator());
              }
            else if (bt || bf)
              {
                jsv.AddMember(
                    JVal().SetString(qs_key.c_str(), jd.GetAllocator()),
                    JVal(bt ? true : false), jd.GetAllocator());
              }
            else
              jsv.AddMember(
                  JVal().SetString(qs_key.c_str(), jd.GetAllocator()),
                  JVal(atoi(qs_value.c_str())), jd.GetAllocator());
          }
      }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    jsv.Accept(writer);
    return buffer.GetString();
  }

  std::shared_ptr<oatpp::web::protocol::http::outgoing::Response>
  OatppJsonAPI::jdoc_to_response(
      const std::shared_ptr<
          oatpp::web::server::api::ApiController::IncomingRequest> &request,
      const JDoc &janswer,
      const std::chrono::time_point<std::chrono::steady_clock> &req_start_time)
  {
    int outcode = janswer["status"]["code"].GetInt();
    std::string stranswer;
    // if output template, fillup with rendered template.
    if (janswer.HasMember("template"))
      {
        std::string tpl = janswer["template"].GetString();
        std::stringstream sg;
        mustache::RenderTemplate(tpl, " ", janswer, &sg);
        stranswer = sg.str();
      }
    else
      {
        stranswer = jrender(janswer);
      }
    if (janswer.HasMember("network"))
      {
        //- grab network call parameters
        std::string url, http_method = "POST",
                         content_type = "Content-Type: application/json";
        if (janswer["network"].HasMember("url"))
          url = janswer["network"]["url"].GetString();
        if (url.empty())
          {
            _logger->error("missing url in network output connector");
            stranswer = jrender(dd_bad_request_400());
            outcode = 400;
          }
        else
          {
            if (janswer["network"].HasMember("http_method"))
              http_method = janswer["network"]["http_method"].GetString();
            if (janswer["network"].HasMember("content_type"))
              content_type = janswer["network"]["content_type"].GetString();

            //- make call
            std::string outstr;
            try
              {
                dd::httpclient::post_call(url, stranswer, http_method, outcode,
                                          outstr, content_type);
                stranswer = outstr;
              }
            catch (std::runtime_error &e)
              {
                _logger->error(e.what());
                _logger->info(stranswer);
                stranswer = jrender(dd_output_connector_network_error_1009());
              }
          }
      }

    // TODO(sileht): START
    // Replace hack by a oatpp ResponseInterceptor when 1.2.5 is released
    if (request)
      {
        auto req = request->getStartingLine();
        std::string access_log = req.protocol.std_str() + " \""
                                 + req.method.std_str() + " "
                                 + req.path.std_str() + "\"";
        if (janswer.HasMember("head") && janswer["head"].HasMember("service"))
          {
            std::string service = janswer["head"]["service"].GetString();
            if (!service.empty())
              access_log += " " + service;
          }
        access_log += " " + std::to_string(outcode);

        auto req_stop_time = std::chrono::steady_clock::now();
        auto req_duration_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                req_stop_time - req_start_time);
        access_log += " " + std::to_string(req_duration_ms.count()) + "ms";

        if (outcode == 200 || outcode == 201)
          _logger->info(access_log);
        else
          _logger->error(access_log);
      }
    // TODO(sileht): END

    auto response = oatpp::web::protocol::http::outgoing::ResponseFactory::
        createResponse(oatpp::web::protocol::http::Status(outcode, ""),
                       stranswer.c_str());
    response->putHeader(oatpp::web::protocol::http::Header::CONTENT_TYPE,
                        "application/json");

    if (!FLAGS_allow_origin.empty())
      response->putHeader("Access-Control-Allow-Origin",
                          oatpp::base::StrBuffer::createFromCString(
                              FLAGS_allow_origin.c_str()));
    return response;
  }

  void OatppJsonAPI::terminate(int signal)
  {
    (void)signal;
    if (_server != nullptr)
      {
        _server->stop();
        _server = nullptr;
      }
  }

#if USE_BOOST_BACKTRACE
  void OatppJsonAPI::abort(int signum)
  {
    std::signal(signum, SIG_DFL);
    std::cerr << boost::stacktrace::stacktrace() << std::endl;
    std::raise(signum);
  }
#endif

  void OatppJsonAPI::run()
  {
    AppComponent components; // Create scope Environment
                             // components

    /* create ApiControllers and add endpoints to router
     */

    auto router = components.httpRouter.getObject();

    auto dedeController = DedeController::createShared(this);
    dedeController->addEndpointsToRouter(router);

    auto docEndpoints = oatpp::swagger::Controller::Endpoints::createShared();
    docEndpoints->pushBackAll(dedeController->getEndpoints());

    auto swaggerController
        = oatpp::swagger::Controller::createShared(docEndpoints);
    swaggerController->addEndpointsToRouter(router);

    auto scp = components.serverConnectionProvider.getObject();
    auto sch = components.serverConnectionHandler.getObject();

    _server = new oatpp::network::Server(scp, sch);

    _logger->info("DeepDetect HTTP server listening on {}:{}",
                  scp->getProperty("host").toString()->c_str(),
                  scp->getProperty("port").toString()->c_str());

    if (!FLAGS_allow_origin.empty())
      _logger->info("Allowing origin from {}", FLAGS_allow_origin);

    std::signal(SIGINT, terminate);
#if USE_BOOST_BACKTRACE
    std::signal(SIGSEGV, abort);
    std::signal(SIGABRT, abort);
#endif
    _server->run();
    _logger->info("DeepDetect HTTP server stopped");
  }

  int OatppJsonAPI::boot(int argc, char *argv[])
  {
    (void)argv;
    (void)argc;
    JsonAPI::boot(argc, argv);

    oatpp::base::Environment::init();

    run();

    oatpp::base::Environment::destroy();
    return 0;
  }
}
