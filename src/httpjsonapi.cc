/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#include <boost/network/protocol/http/server.hpp>
#include <boost/network/uri.hpp>
#include <boost/network/uri/uri_io.hpp>
#include "httpjsonapi.h"
#include "utils/utils.hpp"
#include <algorithm>
#include <iostream>
#include <gflags/gflags.h>

namespace http = boost::network::http;
namespace uri = boost::network::uri;
class APIHandler;
typedef http::server<APIHandler> http_server;

DEFINE_string(host,"localhost","host for running the server");
DEFINE_string(port,"8080","server port");

class APIHandler
{
public:
  APIHandler(dd::HttpJsonAPI *hja)
    :_hja(hja) { }
  
  ~APIHandler() { }

  void fillup_response(http_server::response &response,
		       const JDoc &janswer)
  {
    int code = janswer["status"]["code"].GetInt();
    std::string stranswer = _hja->jrender(janswer);
    response = http_server::response::stock_reply(http_server::response::status_type(code),stranswer);
    //TODO: 409 not built-in cpp netlib
  }

  void operator()(http_server::request const &request,
		  http_server::response &response)
  {
    /*std::cerr << "uri=" << request.destination << std::endl;
    std::cerr << "method=" << request.method << std::endl;
    std::cerr << "source=" << request.source << std::endl;
    std::cerr << "body=" << request.body << std::endl;*/
    uri::uri ur("http://"+request.source+request.destination);
    
    std::string req_method = request.method;
    std::string req_path = uri::path(ur);
    std::string req_query = uri::query(ur);
    std::transform(req_path.begin(),req_path.end(),req_path.begin(),::tolower);
    std::transform(req_query.begin(),req_query.end(),req_query.begin(),::tolower);
    std::vector<std::string> rscs = dd::dd_utils::split(req_path,'/');
    if (rscs.empty())
      {
	response = http_server::response::stock_reply(http_server::response::not_found,_hja->jrender(_hja->dd_not_found_404()));
	return;
      }
    std::string body = request.body;
    
    //debug
    /*std::cerr << "ur=" << ur << std::endl;
    std::cerr << "path=" << req_path << std::endl;
    std::cerr << "query=" << req_query << std::endl;
    std::cerr << "rscs size=" << rscs.size() << std::endl;
    std::cerr << "path1=" << rscs[1] << std::endl;*/
    LOG(INFO) << "HTTP " << req_method << " / call / uri=" << ur << std::endl;
    //debug
    
    if (rscs.at(0) == _rsc_info)
      {
	fillup_response(response,_hja->info());
      }
    else if (rscs.at(0) == _rsc_services)
      {
	if (rscs.size() < 2)
	  {
	    fillup_response(response,_hja->dd_bad_request_400());
	    return;
	  }
	std::string sname = rscs.at(1);
	std::cerr << "sname=" << sname << std::endl;
	if (req_method == "GET")
	  {
	    fillup_response(response,_hja->service_status(sname));
	  }
	else if (req_method == "PUT" || req_method == "POST") // tolerance to using POST
	  {
	    fillup_response(response,_hja->service_create(sname,body));
	  }
	else if (req_method == "DELETE")
	  {
	    // DELETE does not accept body so query options are turned into JSON for internal processing
	    std::string jstr;
	    if (!req_query.empty())
	      {
		std::vector<std::string> vclear = dd::dd_utils::split(req_query,'=');
		if (vclear.size() == 2)
		  jstr = "{\"" + vclear.at(0) + "\":\"" + vclear.at(1) + "\"}";
	      }
	    fillup_response(response,_hja->service_delete(sname,jstr));
	  }
      }
    else if (rscs.at(0) == _rsc_predict)
      {
	if (req_method != "POST")
	  {
	    fillup_response(response,_hja->dd_bad_request_400());
	    return;
	  }
	fillup_response(response,_hja->service_predict(body));
      }
    else if (rscs.at(0) == _rsc_train)
      {
	if (req_method == "GET")
	  {
	    fillup_response(response,_hja->service_train_status(body));
	  }
	else if (req_method == "PUT" || req_method == "POST")
	  {
	    fillup_response(response,_hja->service_train(body));
	  }
	else if (req_method == "DELETE")
	  {
	    // DELETE does not accept body so query options are turned into JSON for internal processing
	    std::string jstr;
	    if (!req_query.empty())
	      {
		jstr = "{";
		std::vector<std::string> voptions = dd::dd_utils::split(req_query,'&');
		for (const std::string o: voptions)
		  {
		    std::vector<std::string> vopt = dd::dd_utils::split(o,'=');
		    if (vopt.size() == 2)
		      {
			if (!jstr.empty())
			  jstr += ",";
			jstr += "\"" + vopt.at(0) + "\":\"" + vopt.at(1) + "\"";
		      }
		  }
		jstr += "}";
	      }
	    std::cout << "jstr=" << jstr << std::endl;
	    fillup_response(response,_hja->service_train_delete(jstr));
	  }
      }
    else
      {
	LOG(ERROR) << "Unknown Service=" << rscs.at(0) << std::endl;
	response = http_server::response::stock_reply(http_server::response::not_found,_hja->jrender(_hja->dd_not_found_404()));
      }
  }
  
  void log(http_server::string_type const &info)
  {
    std::cerr << "ERROR: " << info << "\n"; //TODO
  }

  dd::HttpJsonAPI *_hja;
  std::string _rsc_info = "info";
  std::string _rsc_services = "services";
  std::string _rsc_predict = "predict";
  std::string _rsc_train = "train";
};

namespace dd
{
  HttpJsonAPI::HttpJsonAPI()
    :JsonAPI()
  {
  }

  HttpJsonAPI::~HttpJsonAPI()
  {
  }
  
  int HttpJsonAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    APIHandler ahandler(this);
    http_server::options options(ahandler);
    http_server dd_server(options.address(FLAGS_host)
			  .port(FLAGS_port));
    LOG(INFO) << "Running DeepDetect HTTP server on " << FLAGS_host << ":" << FLAGS_port << std::endl;
    dd_server.run();

    //TODO: capture Ctrl-C signal

    return 0;
  }

}
