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

#include "httpjsonapi.h"
#include "utils/utils.hpp"
#include <algorithm>
#include <csignal>
#include <iostream>
#include "ext/rmustache/mustache.h"
#include "ext/rapidjson/document.h"
#include "ext/rapidjson/stringbuffer.h"
#include "ext/rapidjson/reader.h"
#include "ext/rapidjson/writer.h"
#include <gflags/gflags.h>
#include "utils/httpclient.hpp"

DEFINE_string(host,"localhost","host for running the server");
DEFINE_string(port,"8080","server port");
DEFINE_int32(nthreads,10,"number of HTTP server threads");

namespace dd
{
  std::string uri_query_to_json(const std::string &req_query)
  {
    if (!req_query.empty())
      {
	JDoc jd;
	JVal jsv(rapidjson::kObjectType);
	jd.SetObject();
	std::vector<std::string> voptions = dd::dd_utils::split(req_query,'&');
	for (const std::string o: voptions)
	  {
	    std::vector<std::string> vopt = dd::dd_utils::split(o,'=');
	    if (vopt.size() == 2)
	      {
		bool is_word = false;
		for (size_t i=0;i<vopt.at(1).size();i++)
		  {
		    if (isalpha(vopt.at(1)[i]))
		      {
			is_word = true;
			break;
		      }
		  }
		// we break '.' into JSON sub-objects
		std::vector<std::string> vpt = dd::dd_utils::split(vopt.at(0),'.');
		JVal jobj(rapidjson::kObjectType);
		if (vpt.size() > 1)
		  {
		    bool bt = dd::dd_utils::iequals(vopt.at(1),"true");
		    bool bf = dd::dd_utils::iequals(vopt.at(1),"false");
		    if (is_word && !bt && !bf)
		      {
			jobj.AddMember(JVal().SetString(vpt.back().c_str(),jd.GetAllocator()),JVal().SetString(vopt.at(1).c_str(),jd.GetAllocator()),jd.GetAllocator());
		      }
		    else if (bt || bf)
		      {
			jobj.AddMember(JVal().SetString(vpt.back().c_str(),jd.GetAllocator()),JVal(bt ? true : false),jd.GetAllocator());
		      }
		    else jobj.AddMember(JVal().SetString(vpt.back().c_str(),jd.GetAllocator()),JVal(atoi(vopt.at(1).c_str())),jd.GetAllocator());
		    for (int b=vpt.size()-2;b>0;b--)
		      {
			JVal jnobj(rapidjson::kObjectType);
			jobj = jnobj.AddMember(JVal().SetString(vpt.at(b).c_str(),jd.GetAllocator()),jobj,jd.GetAllocator());
		      }
		    jsv.AddMember(JVal().SetString(vpt.at(0).c_str(),jd.GetAllocator()),jobj,jd.GetAllocator());
		  }
		else
		  {
		    bool bt = dd::dd_utils::iequals(vopt.at(1),"true");
		    bool bf = dd::dd_utils::iequals(vopt.at(1),"false");
		    if (is_word && !bt && !bf)
		      {
			jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),jd.GetAllocator()),JVal().SetString(vopt.at(1).c_str(),jd.GetAllocator()),jd.GetAllocator());
		      }
		    else if (bt || bf)
		      {
			jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),jd.GetAllocator()),JVal(bt ? true : false),jd.GetAllocator());
		      }
		    else jsv.AddMember(JVal().SetString(vopt.at(0).c_str(),jd.GetAllocator()),JVal(atoi(vopt.at(1).c_str())),jd.GetAllocator());
		  }
	      }
	  }
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	jsv.Accept(writer);
	return buffer.GetString();
      }
    else return "";
  }
}

class APIHandler
{
public:
  APIHandler(dd::HttpJsonAPI *hja)
    :_hja(hja) { }
  
  ~APIHandler() { }

  /*static boost::network::http::basic_response<std::string> cstock_reply(int status,
									std::string const& content) {
    using boost::lexical_cast;
    boost::network::http::basic_response<std::string> rep;
    rep.status = status;
    rep.content = content;
    rep.headers.resize(2);
    rep.headers[0].name = "Content-Length";
    rep.headers[0].value = lexical_cast<std::string>(rep.content.size());
    rep.headers[1].name = "Content-Type";
    rep.headers[1].value = "text/html";
    return rep;
    }*/
  
  void fillup_response(http_server::response &response,
		       const JDoc &janswer)
  {
    int code = janswer["status"]["code"].GetInt();
    int outcode = code;
    std::string stranswer;
    if (janswer.HasMember("template")) // if output template, fillup with rendered template.
      {
	std::string tpl = janswer["template"].GetString();
	std::stringstream sg;
	mustache::RenderTemplate(tpl," ",janswer,&sg);
	stranswer = sg.str();
      }
    else
      {
	stranswer = _hja->jrender(janswer);
      }
    if (janswer.HasMember("network"))
      {
	//- grab network call parameters
	std::string url,http_method="POST",content_type="Content-Type: application/json";
	if (janswer["network"].HasMember("url"))
	  url = janswer["network"]["url"].GetString();
	if (url.empty())
	  {
	    LOG(ERROR) << "missing url in network output connector";
	    stranswer = _hja->jrender(_hja->dd_bad_request_400());
	    code = 400;
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
		dd::httpclient::post_call(url,stranswer,http_method,outcode,outstr,content_type);
		stranswer = outstr;
	      }
	    catch (std::runtime_error &e)
	      {
		LOG(ERROR) << e.what() << std::endl;
		LOG(INFO) << stranswer << std::endl;
		stranswer = _hja->jrender(_hja->dd_output_connector_network_error_1009());
	      }
	  }
      }
    response = http_server::response::stock_reply(http_server::response::status_type(outcode),stranswer);
    response.status = static_cast<http_server::response::status_type>(code);
  }

  void operator()(http_server::request const &request,
		  http_server::response &response)
  {
    //debug
    /*std::cerr << "uri=" << request.destination << std::endl;
    std::cerr << "method=" << request.method << std::endl;
    std::cerr << "source=" << request.source << std::endl;
    std::cerr << "body=" << request.body << std::endl;*/
    //debug

    std::string source = request.source;
    if (source == "::1")
      source = "127.0.0.1";
    uri::uri ur;
    ur << uri::scheme("http")
       << uri::host(source)
       << uri::path(request.destination);

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
	    std::string jstr = dd::uri_query_to_json(req_query);
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
	    std::string jstr = dd::uri_query_to_json(req_query);
	    fillup_response(response,_hja->service_train_status(jstr));
	  }
	else if (req_method == "PUT" || req_method == "POST")
	  {
	    fillup_response(response,_hja->service_train(body));
	  }
	else if (req_method == "DELETE")
	  {
	    // DELETE does not accept body so query options are turned into JSON for internal processing
	    std::string jstr = dd::uri_query_to_json(req_query);
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

  /* variables for C-like signal handling */
  HttpJsonAPI *_ghja = nullptr;
  http_server *_gdd_server = nullptr;
  
  HttpJsonAPI::HttpJsonAPI()
    :JsonAPI()
  {
  }

  HttpJsonAPI::~HttpJsonAPI()
  {
    if (_dd_server)
      delete _dd_server;
  }

  int HttpJsonAPI::start_server(const std::string &host,
				const std::string &port,
				const int &nthreads)
  {
    APIHandler ahandler(this);
    http_server::options options(ahandler);
    _dd_server = new http_server(options.address(host)
				 .port(port)
				 .linger(false)
				 .reuse_address(true));
    _ghja = this;
    _gdd_server = _dd_server;
    std::cerr << "Running DeepDetect HTTP server on " << host << ":" << port << std::endl;

    std::vector<std::thread> ts;
    for (int i=0;i<nthreads;i++)
      ts.push_back(std::thread(std::bind(&http_server::run,_dd_server)));
    try {
      _dd_server->run();
    }
    catch(std::exception &e)
      {
	LOG(ERROR) << e.what() << std::endl;
	return 1;
      }
    for (int i=0;i<nthreads;i++)
      ts.at(i).join();
    return 0;
  }

  int HttpJsonAPI::start_server_daemon(const std::string &host,
				       const std::string &port,
				       const int &nthreads)
  {
    //_ft = std::async(&HttpJsonAPI::start_server,this,host,port,nthreads);
    std::thread t(&HttpJsonAPI::start_server,this,host,port,nthreads);
    t.detach();
    return 0;
  }
  
  void HttpJsonAPI::stop_server()
  {
    LOG(INFO) << "stopping HTTP server\n";
    if (_dd_server)
      {
	try
	  {
	    _dd_server->stop();
	    _ft.wait();
	    delete _dd_server;
	    _gdd_server = nullptr;
	  }
	catch (std::exception &e)
	  {
	    LOG(ERROR) << e.what() << std::endl;
	  }
      }
  }

  void HttpJsonAPI::terminate(int param)
   {
    (void)param;
    if (_ghja)
      {
	_ghja->stop_server();
      }
   }
   
  int HttpJsonAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::signal(SIGINT,terminate);
    return start_server(FLAGS_host,FLAGS_port,FLAGS_nthreads);
  }

}
