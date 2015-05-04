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
#include <csignal>
#include <iostream>
#include <gflags/gflags.h>

namespace http = boost::network::http;
namespace uri = boost::network::uri;
class APIHandler;
typedef http::server<APIHandler> http_server;

DEFINE_string(host,"localhost","host for running the server");
DEFINE_string(port,"8080","server port");
DEFINE_int32(nthreads,10,"number of HTTP server threads");

class APIHandler
{
public:
  APIHandler(dd::HttpJsonAPI *hja)
    :_hja(hja) { }
  
  ~APIHandler() { }

  static std::string uri_query_to_json(const std::string &req_query)
  {
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
		if (jstr != "{")
		  jstr += ",";
		jstr += "\"" + vopt.at(0) + "\":";
		bool is_word = true;
		for (size_t i=0;i<vopt.at(1).size();i++)
		  {
		    if (isalpha(vopt.at(1)[i]) == 0)
		      {
			is_word = false;
			break;
		      }
		  }
		if (is_word) 
		  jstr += "\"" + vopt.at(1) + "\"";
		else jstr += vopt.at(1);
	      }
	  }
	jstr += "}";
      }
    std::cout << "jstr=" << jstr << std::endl;
    return jstr;
  }
  
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
	LOG(ERROR) << "empty resource\n";
	response = http_server::response::stock_reply(http_server::response::not_found,_hja->jrender(_hja->dd_not_found_404()));
	return;
      }
    std::string body = request.body;
    
    //debug
    /*std::cerr << "ur=" << ur << std::endl;
    std::cerr << "path=" << req_path << std::endl;
    std::cerr << "query=" << req_query << std::endl;
    std::cerr << "rscs size=" << rscs.size() << std::endl;
    std::cerr << "path1=" << rscs[1] << std::endl;
    LOG(INFO) << "HTTP " << req_method << " / call / uri=" << ur << std::endl;*/
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
	    std::string jstr = uri_query_to_json(req_query);
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
	    std::string jstr = uri_query_to_json(req_query);
	    fillup_response(response,_hja->service_train_status(jstr));
	  }
	else if (req_method == "PUT" || req_method == "POST")
	  {
	    fillup_response(response,_hja->service_train(body));
	  }
	else if (req_method == "DELETE")
	  {
	    // DELETE does not accept body so query options are turned into JSON for internal processing
	    std::string jstr = uri_query_to_json(req_query);
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
    LOG(ERROR) << info << std::endl;
  }

  dd::HttpJsonAPI *_hja;
  std::string _rsc_info = "info";
  std::string _rsc_services = "services";
  std::string _rsc_predict = "predict";
  std::string _rsc_train = "train";
};

namespace dd
{
  volatile std::sig_atomic_t _sigstatus;
  http_server *_dd_server;
  
  void sig_handler(int signal)
  {
    _sigstatus = signal;
    LOG(INFO) << "catching termination signal " << _sigstatus << std::endl;;
    _dd_server->stop(); // stops acceptor and waits for pending requests to finish
    exit(1); // beware, does not cleanly kill all jobs, async jobs should be killed cleanly through API
  }
  
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
    std::signal(SIGINT,sig_handler);
    
    APIHandler ahandler(this);
    http_server::options options(ahandler);
    http_server dd_server(options.address(FLAGS_host)
			  .port(FLAGS_port));
    _dd_server = &dd_server;
    LOG(INFO) << "Running DeepDetect HTTP server on " << FLAGS_host << ":" << FLAGS_port << std::endl;

    std::vector<std::thread> ts;
    for (int i=0;i<FLAGS_nthreads;i++)
      ts.push_back(std::thread(std::bind(&http_server::run,&dd_server)));
    dd_server.run();
    for (int i=0;i<FLAGS_nthreads;i++)
      ts.at(i).join();
    return 0;
  }

}
