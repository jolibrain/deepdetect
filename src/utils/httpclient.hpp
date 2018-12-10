/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
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

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>

#ifndef DD_HTTPCLIENT_H
#define DD_HTTPCLIENT_H

namespace dd
{

  class httpclient
  {
  public:
    static void get_call(const std::string &url,
			 const std::string &http_method,
			 int &outcode,
			 std::string &outstr)
    {
      curlpp::Cleanup cl;
      std::ostringstream os;
      curlpp::Easy request;
      curlpp::options::WriteStream ws(&os);
      curlpp::options::CustomRequest pr(http_method);
      request.setOpt(curlpp::options::Url(url));
      request.setOpt(ws);
      request.setOpt(pr);
      request.setOpt(cURLpp::Options::FollowLocation(true));
      request.perform();
      outstr = os.str();
      //std::cout << "outstr=" << outstr << std::endl;
      outcode = curlpp::infos::ResponseCode::get(request);
    }
    
    static void post_call(const std::string &url,
			  const std::string &jcontent,
			  const std::string &http_method,
			  int &outcode,
			  std::string &outstr,
			  const std::string &content_type="Content-Type: application/json")
    {
      curlpp::Cleanup cl;
      std::ostringstream os;
      curlpp::Easy request_put;
      curlpp::options::WriteStream ws(&os);
      curlpp::options::CustomRequest pr(http_method);
      request_put.setOpt(curlpp::options::Url(url));
      request_put.setOpt(ws);
      request_put.setOpt(pr);
      std::list<std::string> header;
      header.push_back(content_type);
      request_put.setOpt(curlpp::options::HttpHeader(header));
      request_put.setOpt(curlpp::options::PostFields(jcontent));
      request_put.setOpt(curlpp::options::PostFieldSize(jcontent.length()));
      request_put.perform();
      outstr = os.str();
      //std::cout << "outstr=" << outstr << std::endl;
      outcode = curlpp::infos::ResponseCode::get(request_put);
    }
    
  };
  
}

#endif
