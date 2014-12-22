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
DEFINE_bool(predict,false,"run service in predict mode");
DEFINE_bool(train,false,"run service in train mode");
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
	    APIData model_ad;
	    model_ad.add("repository",FLAGS_model_repo);
	    CaffeModel cmodel(model_ad);
	    add_service(FLAGS_service,std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(FLAGS_service,cmodel)));
	  }
	if (!FLAGS_imgfname.empty())
	  {
	    std::cout << FLAGS_imgfname << std::endl;
	    APIData ad;
	    //ad.add("imgfname",FLAGS_imgfname);
	    std::vector<std::string> vdata = { FLAGS_imgfname };
	    ad.add("data",vdata);
	    APIData out;
	    predict(ad,0,out);
	    //std::cout << "witness=\n" << out.to_str() << std::endl;
	    //std::string tpl = "status={{status}}\n{{cat0}} --> {{prob0}}\n{{cat1}} --> {{prob1}}\n{{cat2}} --> {{prob2}}\n";
	    std::string tpl = "status={{status}}\n{{# predictions}}loss={{loss}}\n{{# classes}}{{cat}} --> {{prob}}\n{{/classes}}{{/predictions}}\n";
	    //std::string tpl = "{{# predictions}}\nloss --> {{loss}}\n{{/ predictions}}\n";
	    std::cout << "response=\n" << out.render_template(tpl) << std::endl;//,"predictions") << std::endl;
	    //APIData pred = out.get("predictions").get<std::vector<APIData>>().at(0);
	    //std::cout << "response=\n" << pred.render_template(tpl) << std::endl;
	    remove_service(FLAGS_service);
	  }
      }
    else if (FLAGS_train)
      {
	if (FLAGS_service == "caffe")
	  {
	    APIData model_ad;
	    model_ad.add("repository",FLAGS_model_repo);
	    CaffeModel cmodel(model_ad);
	    add_service(FLAGS_service,std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(FLAGS_service,cmodel)));
	    APIData ad, out;
	    train(ad,0,out);
	    std::string tpl = "status={{status}}\n";
	    std::cout << "response=\n" << out.render_template(tpl) << std::endl;
	    remove_service(FLAGS_service);
	  }
      }

    return 0;
  }
  
}
