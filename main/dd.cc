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
#include "caffelib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

using namespace dd;

DEFINE_bool(jsonapi,false,"whether to use the JSON command line API");

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_jsonapi)
    {
      DeepDetect<CommandLineAPI> dd;
      dd.boot(argc,argv);
    }
  else
    {
      DeepDetect<CommandLineJsonAPI> dd;
      dd.boot(argc,argv);
    }
}
