/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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

#include <iostream>

#include "deepdetect.h"
#ifdef USE_COMMAND_LINE
#ifdef USE_CAFFE
#include "commandlineapi.h"
#endif // USE_CAFFE
#ifdef USE_JSON_API
#include "commandlinejsonapi.h"
#endif // USE_JSON_API
#endif // USE_COMMAND_LINE
#ifdef USE_HTTP_SERVER
#include "httpjsonapi.h"
#endif
#ifdef USE_HTTP_SERVER_OATPP
#include "oatppjsonapi.h"
#endif
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"
#ifdef USE_XGBOOST
#include <rabit/rabit.h>
#endif
#include <gflags/gflags.h>

using namespace dd;

DEFINE_int32(jsonapi, 0,
             "whether to use the JSON command line API ("
#if defined(USE_HTTP_SERVER) || defined(USE_HTTP_SERVER_OATPP)
             "0: HTTP server JSON "
#endif // USE_HTTP_SERVER
#ifdef USE_COMMAND_LINE
#ifdef USE_JSON_API
             "1: commandline JSON  "
#endif // USE_JSON_API
#ifdef USE_CAFFE
             "2: commandline no JSON  "
#endif // USE_CAFFE
#endif // USE_COMMAND_LINE
#ifdef USE_HTTP_SERVER_OATPP
             "3: HTTP server JSON (oat++ version)"
#endif // USE_HTTP_SERVER_OATPP
#ifdef USE_HTTP_SERVER
             "4: HTTP server JSON (cppnetlib version)"
#endif // USE_HTTP_SERVER

);

void invalid_jsonapi_flag(const std::string option)
{
  std::cerr << "--jsonapi=" << FLAGS_jsonapi
            << " is unavailable. deepdetect has been compiled without "
            << option << std::endl;
}

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
#ifdef USE_XGBOOST
  rabit::Init(argc, argv);
#endif // USE_XGBOOST

  switch (FLAGS_jsonapi)
    {
    case 0:
#if defined(USE_HTTP_SERVER) || defined(USE_HTTP_SERVER_OATPP)
      {
#ifdef USE_HTTP_SERVER
        DeepDetect<HttpJsonAPI> dd;
#else
        DeepDetect<OatppJsonAPI> dd;
#endif // USE_HTTP_SERVER_OATPP
        dd.boot(argc, argv);
      }
      break;
#else
      invalid_jsonapi_flag("USE_HTTP_SERVER OR USE_HTTP_SERVER_OATPP");
      return 1;
#endif // USE_HTTP_SERVER OR USE_HTTP_SERVER_OATPP

    case 1:
#if defined(USE_JSON_API) && defined(USE_COMMANDLINE)
      {
        DeepDetect<CommandLineJsonAPI> dd;
        dd.boot(argc, argv);
      }
      break;
#else // USE_JSON_API
      invalid_jsonapi_flag("USE_JSON_API and USE_COMMANDLINE");
      return 1;
#endif // USE_JSON_API

    case 2:
#if defined(USE_CAFFE) && defined(USE_COMMANDLINE)
      {
        DeepDetect<CommandLineAPI> dd;
        dd.boot(argc, argv);
      }
      break;
#else // USE_CAFFE
      invalid_jsonapi_flag("USE_CAFFE and USE_COMMANDLINE");
      return 1;
#endif // USE_CAFFE

    case 3:
#ifdef USE_HTTP_SERVER_OATPP
      {
        DeepDetect<OatppJsonAPI> dd;
        dd.boot(argc, argv);
      }
      break;
#else // USE_HTTP_SERVER_OATPP
      invalid_jsonapi_flag("USE_HTTP_SERVER_OATPP");
      return 1;
#endif // USE_HTTP_SERVER_OATPP

    case 4:
#ifdef USE_HTTP_SERVER
      {
        DeepDetect<HttpJsonAPI> dd;
        dd.boot(argc, argv);
      }
      break;
#else // USE_HTTP_SERVER
      invalid_jsonapi_flag("USE_HTTP_SERVER");
      return 1;
#endif // USE_HTTP_SERVER

    default:
      std::cerr << "--jsonapi value is invalid" << std::endl;
      return 1;
    }
#ifdef USE_XGBOOST
  rabit::Finalize();
#endif
  return 0;
}
