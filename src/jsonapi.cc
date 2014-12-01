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

#include "jsonapi.h"
#include "ext/rapidjson/document.h"
#include "ext/rapidjson/stringbuffer.h"
#include "ext/rapidjson/reader.h"
#include "ext/rapidjson/writer.h"
#include <glog/logging.h>

//using namespace rapidjson;

namespace dd
{
  
  JsonAPI::JsonAPI()
    :APIStrategy()
  {
  }

  JsonAPI::~JsonAPI()
  {
  }

  int JsonAPI::boot(int argc, char *argv[])
  {
    return 0; // does nothing, in practice, class should be derived.
  }

  void JsonAPI::render_status(JDoc &jst,
			      const uint32_t &code, const std::string &msg,
			      const uint32_t &dd_code, const std::string &dd_msg) const
  {
    JVal jsv(rapidjson::kObjectType);
    jsv.AddMember("code",JVal(code).Move(),jst.GetAllocator());
    if (!msg.empty())
      jsv.AddMember("msg",JVal().SetString(msg.c_str(),jst.GetAllocator()),jst.GetAllocator());
    if (dd_code > 0)
      {
	jsv.AddMember("dd_code",JVal(dd_code).Move(),jst.GetAllocator());
	if (!dd_msg.empty())
	  jsv.AddMember("dd_msg",JVal().SetString(dd_msg.c_str(),jst.GetAllocator()),jst.GetAllocator());
      }
    jst.AddMember("status",jsv,jst.GetAllocator());
  }

  JDoc JsonAPI::dd_ok_200() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,200,"OK");
    return jd;
  }

  JDoc JsonAPI::dd_created_201() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,201,"Created");
    return jd;
  }

  JDoc JsonAPI::dd_bad_request_400() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"BadRequest");
    return jd;
  }
  
  JDoc JsonAPI::dd_forbidden_403() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"Forbidden");
    return jd;
  }

  JDoc JsonAPI::dd_not_found_404() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"NotFound");
    return jd;
  }

  JDoc JsonAPI::dd_internal_error_500() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,500,"InternalError");
    return jd;
  }
  
  JDoc JsonAPI::dd_unknown_library_1000() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"NotFound",1000,"Unknown Library");
    return jd;
  }

  JDoc JsonAPI::dd_no_data_1001() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"BadRequest",1001,"No Data");
    return jd;
  }
  
  std::string JsonAPI::jrender(const JDoc &jst) const
  {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    jst.Accept(writer);
    return buffer.GetString();
  }

  std::string JsonAPI::jrender(const JVal &jval) const
  {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    jval.Accept(writer);
    return buffer.GetString();
  }

  std::string JsonAPI::info() const
  {
    // answer info call.
    JDoc jinfo = dd_ok_200();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/info",jinfo.GetAllocator());
    //TODO: server version.
    JVal jservs(rapidjson::kArrayType);
    for (size_t i=0;i<services_size();i++)
      {
	APIData ad = mapbox::util::apply_visitor(visitor_info(),_mlservices.at(i));
	JVal jserv(rapidjson::kObjectType);
	ad.toJVal(jinfo,jserv);
	jservs.PushBack(jserv,jinfo.GetAllocator());
      }
    jhead.AddMember("services",jservs,jinfo.GetAllocator());
    jinfo.AddMember("head",jhead,jinfo.GetAllocator());
    return jrender(jinfo);
  }

  std::string JsonAPI::service_create(const std::string &sname,
				      const std::string &jstr)
  {
    if (sname.empty())
      return jrender(dd_not_found_404());
    
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      return jrender(dd_bad_request_400());
    
    std::string mllib = d["mllib"].GetString();
    if (mllib.empty())
      return jrender(dd_bad_request_400());
    std::string description = d["description"].GetString();

    // model parameters.
    APIData ad_model(d["model"]);
    
    // create service.
    if (mllib == "caffe")
      {
	CaffeModel cmodel(ad_model);
	add_service(sname,std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)));
      }
    else
      {
	return jrender(dd_unknown_library_1000());
      }

    JDoc jsc = dd_created_201();
    return jrender(jsc);
  }
  
  std::string JsonAPI::service_status(const std::string &sname)
  {
    if (sname.empty())
      return jrender(dd_not_found_404());
    int pos = this->get_service_pos(sname);
    if (pos < 0)
      return jrender(dd_not_found_404());
    APIData ad = mapbox::util::apply_visitor(visitor_status(),_mlservices.at(pos));
    JDoc jst = dd_ok_200();
    ad.toJDoc(jst);
    return jrender(jst);
  }

  std::string JsonAPI::service_delete(const std::string &sname)
  {
    if (sname.empty())
      return jrender(dd_not_found_404());
    if (remove_service(sname))
      return jrender(dd_ok_200());
    return jrender(dd_not_found_404());
  }

  std::string JsonAPI::service_predict(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      return jrender(dd_bad_request_400());
    
    // service
    std::string sname = d["service"].GetString();
    int pos = this->get_service_pos(sname);
    if (pos < 0)
      return jrender(dd_not_found_404());

    // data
    APIData ad_data(d);
  
    // prediction
    APIData out;
    int status = this->predict(ad_data,pos,out);
    if (status < 0)
      return jrender(dd_internal_error_500());
    JDoc jpred = dd_ok_200();
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jpred,jout);
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/predict",jpred.GetAllocator());
    jhead.AddMember("time",jout["time"],jpred.GetAllocator());
    jhead.AddMember("service",d["service"],jpred.GetAllocator());
    jpred.AddMember("head",jhead,jpred.GetAllocator());
    jpred.AddMember("predictions",jout["predictions"],jpred.GetAllocator());
    return jrender(jpred);
  }

}
