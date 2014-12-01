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

#ifndef JSONAPI_H
#define JSONAPI_H

#include "apistrategy.h"
#include "dd_types.h"

namespace dd
{
  class JsonAPI : public APIStrategy
  {
  public:
    JsonAPI();
    ~JsonAPI();

    int boot(int argc, char *argv[]);

    // error status
    void render_status(JDoc &jst,
		       const uint32_t &code, const std::string &msg,
		       const uint32_t &dd_code=0, const std::string &dd_msg="") const;
    
    // errors
    JDoc dd_ok_200() const;
    JDoc dd_created_201() const;
    JDoc dd_bad_request_400() const;
    JDoc dd_forbidden_403() const;
    JDoc dd_not_found_404() const;

    // specific errors
    JDoc dd_unknown_library_1000() const;
    JDoc dd_no_data_1001() const;

    // JSON rendering
    std::string jrender(const JDoc &jst) const;
    
    // resources
    std::string info() const;
    
    std::string service_create(const std::string &sname, const std::string &jstr);
    std::string service_status(const std::string &sname);
    std::string service_delete(const std::string &sname);
  };

  class visitor_info : public mapbox::util::static_visitor<APIData>
  {
  public:
    visitor_info() {}
    ~visitor_info() {}

    template<typename T>
      APIData operator() (T &mllib)
      {
	return mllib.info();
      }
  };

  class visitor_status : public mapbox::util::static_visitor<APIData>
  {
  public:
    visitor_status() {}
    ~visitor_status() {}

    template<typename T>
      APIData operator() (T &mllib)
      {
	return mllib.status();
      }
  };

}

#endif
