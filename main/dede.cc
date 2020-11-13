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
#ifdef USE_HTTP_SERVER
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

);

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
#ifdef USE_XGBOOST
  rabit::Init(argc, argv);
#endif // USE_XGBOOST

#ifdef USE_HTTP_SERVER
  if (FLAGS_jsonapi == 0)
    {
      DeepDetect<HttpJsonAPI> dd;
      dd.boot(argc, argv);
    }
#endif // USE_HTTP_SERVER
#ifdef USE_COMMANDLINE
#ifdef USE_CAFFE
  if (FLAGS_jsonapi == 2)
    {
      DeepDetect<CommandLineAPI> dd;
      dd.boot(argc, argv);
    }
#endif // USE_CAFFE
#ifdef USE_JSON_API
  if (FLAGS_jsonapi == 1)
    {
      DeepDetect<CommandLineJsonAPI> dd;
      dd.boot(argc, argv);
    }
#endif // USE_JSON_API
#endif // USE_COMMANDLINE

#ifdef USE_HTTP_SERVER_OATPP
  if (FLAGS_jsonapi == 3)
    {
      DeepDetect<OatppJsonAPI> dd;
      dd.boot(argc, argv);
    }
#endif // USE_HTTP_SERVER_OATPP

#ifdef USE_XGBOOST
  rabit::Finalize();
#endif
}
