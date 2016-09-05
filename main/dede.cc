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
#include "commandlineapi.h"
#include "commandlinejsonapi.h"
#include "httpjsonapi.h"
#include "caffelib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"
#ifdef USE_XGBOOST
#include <rabit/rabit.h>
#endif

using namespace dd;

DEFINE_int32(jsonapi,0,"whether to use the JSON command line API (0: HTTP server JSON, 1: commandline JSON, 2: commandline no JSON");

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  //::google::InitGoogleLogging(argv[0]);
#ifdef USE_XGBOOST
  rabit::Init(argc,argv);
#endif
  
  if (FLAGS_jsonapi == 0)
    {
      DeepDetect<HttpJsonAPI> dd;
      dd.boot(argc,argv);
    }
  else if (FLAGS_jsonapi == 2)
    {
      DeepDetect<CommandLineAPI> dd;
      dd.boot(argc,argv);
    }
  else if (FLAGS_jsonapi == 1)
    {
      DeepDetect<CommandLineJsonAPI> dd;
      dd.boot(argc,argv);
    }
#ifdef USE_XGBOOST
  rabit::Finalize();
#endif
}
