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

#include "commandlineapi.h"
#include <gflags/gflags.h>
#include <iostream>

DEFINE_string(service,"","service string (e.g. caffe)");
DEFINE_bool(predict,true,"run service in predict mode");
DEFINE_string(model_repo,"","model repository");
DEFINE_string(imgfname,"","image file name");

namespace dd
{

  CommandLineAPI::CommandLineAPI()
    :APIStrategy()
  {
  }

  CommandLineAPI::~CommandLineAPI()
  {
  }

  int CommandLineAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    // create service.
    if (FLAGS_service.empty())
      {
	std::cout << "service required\n";
	return 1;
      }

    if (FLAGS_predict)
      {
	if (FLAGS_service == "caffe")
	  {
	    CaffeModel cmodel = CaffeModel::read_from_repository(FLAGS_model_repo);
	    add_service(std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(cmodel)));
	  }
      }
    if (!FLAGS_imgfname.empty())
      {
	std::cout << FLAGS_imgfname << std::endl;
	APIData ad;
	ad.add("imgfname",FLAGS_imgfname);
	std::string out;
	predict(ad,0,out);
	std::cout << "response=\n" << out << std::endl;
      }

    return 0;
  }
  
}
