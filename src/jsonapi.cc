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
#include "dd_config.h"
#include "githash.h"
#include "ext/rapidjson/document.h"
#include "ext/rapidjson/stringbuffer.h"
#include "ext/rapidjson/reader.h"
#include "ext/rapidjson/writer.h"
#include <glog/logging.h>

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

  JDoc JsonAPI::dd_service_not_found_1002() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"NotFound",1002,"Service Not Found");
    return jd;
  }
  
  JDoc JsonAPI::dd_job_not_found_1003() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"NotFound",1003,"Job Not Found");
    return jd;
  }

  JDoc JsonAPI::dd_input_connector_not_found_1004() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"NotFound",1004,"Input Connector Not Found");
    return jd;
  }

  JDoc JsonAPI::dd_service_input_bad_request_1005() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"BadRequest",1005,"Service Input Error");
    return jd;
  }
  
  JDoc JsonAPI::dd_service_bad_request_1006() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"BadRequest",1006,"Service Bad Request Error");
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
    std::string sversion = std::to_string(VERSION_MAJOR) + "." + std::to_string(VERSION_MINOR);
    jhead.AddMember("version",JVal().SetString(sversion.c_str(),jinfo.GetAllocator()),jinfo.GetAllocator());
    jhead.AddMember("branch",JVal().SetString(GIT_BRANCH,jinfo.GetAllocator()),jinfo.GetAllocator());
    jhead.AddMember("commit",JVal().SetString(GIT_COMMIT_HASH,jinfo.GetAllocator()),jinfo.GetAllocator());
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
    
    std::string mllib,input;
    std::string type,description;
    APIData ad,ad_model;
    try
      {
	// mandatory parameters.
	mllib = d["mllib"].GetString();
	input = d["parameters"]["input"]["connector"].GetString();

	// optional parameters.
	if (d.HasMember("type"))
	  type = d["type"].GetString();
	if (d.HasMember("description"))
	  description = d["description"].GetString();
	
	// model parameters (mandatory).
	ad = APIData(d);
	ad_model = ad.getobj("model");//APIData(d["model"]);
      }
    catch(...)
      {
	return jrender(dd_bad_request_400());
      }
        
    // create service.
    if (mllib == "caffe")
      {
	CaffeModel cmodel(ad_model);
	if (input == "image")
	  add_service(sname,std::move(MLService<CaffeLib,ImgInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
	else return jrender(dd_input_connector_not_found_1004());
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
      return jrender(dd_service_not_found_1002());
    int pos = this->get_service_pos(sname);
    if (pos < 0)
      return jrender(dd_not_found_404());
    APIData ad = mapbox::util::apply_visitor(visitor_status(),_mlservices.at(pos));
    JDoc jst = dd_ok_200();
    JVal jbody(rapidjson::kObjectType);
    ad.toJVal(jst,jbody);
    jst.AddMember("body",jbody,jst.GetAllocator());
    return jrender(jst);
  }

  std::string JsonAPI::service_delete(const std::string &sname)
  {
    if (sname.empty())
      return jrender(dd_service_not_found_1002());
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
    std::string sname;
    int pos = -1;
    try
      {
	sname = d["service"].GetString();
	pos = this->get_service_pos(sname);
	if (pos < 0)
	  return jrender(dd_service_not_found_1002());
      }
    catch(...)
      {
	return jrender(dd_bad_request_400());
      }

    // data
    APIData ad_data(d);
  
    // prediction
    APIData out;
    int status = -1;
    try
      {
	status = this->predict(ad_data,pos,out);
      }
    catch (InputConnectorBadParamException &e)
      {
	return jrender(dd_service_input_bad_request_1005());
      }
    catch (MLLibBadParamException &e)
      {
	return jrender(dd_service_bad_request_1006());
      }
    catch (InputConnectorInternalException &e)
      {
	return jrender(dd_internal_error_500());
      }
    catch (MLLibInternalException &e)
      {
	return jrender(dd_internal_error_500());
      }
    catch (...)
      {
	return jrender(dd_internal_error_500());
      }
    JDoc jpred = dd_ok_200();
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jpred,jout);
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/predict",jpred.GetAllocator());
    jhead.AddMember("time",jout["time"],jpred.GetAllocator());
    jhead.AddMember("service",d["service"],jpred.GetAllocator());
    jpred.AddMember("head",jhead,jpred.GetAllocator());
    JVal jbody(rapidjson::kObjectType);
    jbody.AddMember("predictions",jout["predictions"],jpred.GetAllocator());
    jpred.AddMember("body",jbody,jpred.GetAllocator());
    return jrender(jpred);
  }

  std::string JsonAPI::service_train(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      return jrender(dd_bad_request_400());
  
    // service
    std::string sname;
    int pos = -1;
    try
      {
	sname = d["service"].GetString();
	pos = this->get_service_pos(sname);
	if (pos < 0)
	  return jrender(dd_service_not_found_1002());
      }
    catch(...)
      {
	return jrender(dd_bad_request_400());
      }
    
    // parameters and data
    APIData ad(d);

    // training
    APIData out;
    int status = -1;
    try
      {
	status = this->train(ad,pos,out);
      }
    catch (InputConnectorBadParamException &e)
      {
	return jrender(dd_service_input_bad_request_1005());
      }
    catch (MLLibBadParamException &e)
      {
	return jrender(dd_service_bad_request_1006());
      }
    catch (InputConnectorInternalException &e)
      {
	return jrender(dd_internal_error_500());
      }
    catch (MLLibInternalException &e)
      {
	return jrender(dd_internal_error_500());
      }
    catch (...)
      {
	return jrender(dd_internal_error_500());
      }
    JDoc jtrain = dd_created_201();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/train",jtrain.GetAllocator());
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jtrain,jout);
    if (jout.HasMember("job")) // async job
      {
	jhead.AddMember("job",static_cast<int>(jout["job"].GetDouble()),jtrain.GetAllocator());
	jout.RemoveMember("job");
	jhead.AddMember("status",JVal().SetString(jout["status"].GetString(),jtrain.GetAllocator()),jtrain.GetAllocator());
	jout.RemoveMember("status");
      }
    else
      {
	jhead.AddMember("time",jout["time"].GetDouble(),jtrain.GetAllocator());
	jout.RemoveMember("time");
	jtrain.AddMember("body",jout,jtrain.GetAllocator());
      }
    jtrain.AddMember("head",jhead,jtrain.GetAllocator());
    return jrender(jtrain);
  }

  std::string JsonAPI::service_train_status(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      return jrender(dd_bad_request_400());
    
    // service
    std::string sname;
    int pos = -1;
    try
      {
	sname = d["service"].GetString();
	pos = this->get_service_pos(sname);
	if (pos < 0)
	  return jrender(dd_service_not_found_1002());
      }
    catch(...)
      {
	return jrender(dd_bad_request_400());
      }
    
    // parameters
    APIData ad(d);
    if (!ad.has("job"))
      return jrender(dd_job_not_found_1003());
    
    // training status
    APIData out;
    int status = this->train_status(ad,pos,out);
    JDoc jtrain;
    if (status == 1)
      {
	jtrain = dd_job_not_found_1003();
	JVal jhead(rapidjson::kObjectType);
	jhead.AddMember("method","/train",jtrain.GetAllocator());
	jhead.AddMember("job",static_cast<int>(ad.get("job").get<double>()),jtrain.GetAllocator());
	jtrain.AddMember("head",jhead,jtrain.GetAllocator());
	return jrender(jtrain);
      }
    jtrain = dd_ok_200();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/train",jtrain.GetAllocator());
    jhead.AddMember("job",static_cast<int>(ad.get("job").get<double>()),jtrain.GetAllocator());
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jtrain,jout);
    jhead.AddMember("status",JVal().SetString(jout["status"].GetString(),jtrain.GetAllocator()),jtrain.GetAllocator());
    jhead.AddMember("time",jout["time"].GetDouble(),jtrain.GetAllocator());
    jout.RemoveMember("time");
    jout.RemoveMember("status");
    jtrain.AddMember("head",jhead,jtrain.GetAllocator());
    jtrain.AddMember("body",jout,jtrain.GetAllocator());
    return jrender(jtrain);
  }

  std::string JsonAPI::service_train_delete(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      return jrender(dd_bad_request_400());
  
    // service
    std::string sname;
    int pos = -1;
    try
      {
	sname = d["service"].GetString();
	pos = this->get_service_pos(sname);
	if (pos < 0)
	  return jrender(dd_service_not_found_1002());
      }
    catch(...)
      {
	return jrender(dd_bad_request_400());
      }
    
    // parameters
    APIData ad(d);
    if (!ad.has("job"))
      return jrender(dd_job_not_found_1003());
    
    // delete training job
    APIData out;
    int status = this->train_delete(ad,pos,out);
    JDoc jd;
    if (status == 1)
      {
	jd = dd_job_not_found_1003();
	JVal jhead(rapidjson::kObjectType);
	jhead.AddMember("method","/train",jd.GetAllocator());
	jhead.AddMember("job",static_cast<int>(ad.get("job").get<double>()),jd.GetAllocator());
	jd.AddMember("head",jhead,jd.GetAllocator());
	return jrender(jd);
      }
    jd = dd_ok_200();
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jd,jout);
    jout.AddMember("method","/train",jd.GetAllocator());
    jout.AddMember("job",static_cast<int>(ad.get("job").get<double>()),jd.GetAllocator());
    jd.AddMember("head",jout,jd.GetAllocator());
    return jrender(jd);
  }

}
