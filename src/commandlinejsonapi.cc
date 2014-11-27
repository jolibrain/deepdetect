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

#include "commandlinejsonapi.h"
#include <gflags/gflags.h>
#include <iostream>

DEFINE_bool(info,false,"/info JSON call");
DEFINE_string(service_create,"","/service/service_name call JSON string");
DEFINE_string(service_name,"","service name string for JSON call /service/service_name");

namespace dd
{

  CommandLineJsonAPI::CommandLineJsonAPI()
    :JsonAPI()
  {
  }

  CommandLineJsonAPI::~CommandLineJsonAPI()
  {
  }

  int CommandLineJsonAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (!FLAGS_service_create.empty())
      {
	if (FLAGS_service_name.empty())
	  {
	    //TODO: error.
	  }
	std::string janswer = service_create(FLAGS_service_name,FLAGS_service_create);
	std::cout << janswer << std::endl;
      }
    if (FLAGS_info)
      {
	std::string janswer = info();
	std::cout << janswer << std::endl;
      }

    return 0;
  }

}
