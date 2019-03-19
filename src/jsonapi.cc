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
#include <gflags/gflags.h>

DEFINE_string(service_start_list,"","list of JSON calls to be executed at startup");

namespace dd
{
  std::string JsonAPI::_json_blob_fname = "model.json";
  std::string JsonAPI::_json_config_blob_fname = "config.json";
  
  JsonAPI::JsonAPI()
    :APIStrategy()
  {
  }

  JsonAPI::~JsonAPI()
  {
  }

  int JsonAPI::boot(int argc, char *argv[])
  {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (!FLAGS_service_start_list.empty())
      service_autostart(FLAGS_service_start_list);
    return 0;
  }

  JDoc JsonAPI::service_autostart(const std::string &autostart_file)
  {
    if (autostart_file.empty())
      return 0;
    if (!fileops::file_exists(autostart_file))
      {
	_logger->error("JSON autostart file not found: {}",autostart_file);
	return dd_bad_request_400();
      }
    std::ifstream injsonfile(autostart_file);
    if (!injsonfile.is_open())
      {
	_logger->error("Failed opening JSON autostart file {}",autostart_file);
	return dd_internal_error_500();
      }

    std::vector<JDoc> calls_output;
    std::string line;
    int lines = 0;
    while(std::getline(injsonfile,line))
      {
	// file format is service_create;sname;JSON or service_predict;JSON
	std::vector<std::string> elts = dd_utils::split(line,';');
	std::string api_call = elts.at(0);
	if ((api_call == "service_create" && elts.size() != 3)
	    || (api_call == "service_predict" && elts.size() != 2))
	  {
	    _logger->error("Error parsing autostart JSON file {} line {}: wrong number of elements",autostart_file,lines);
	    return dd_bad_request_400();
	  }
	
	// dispatch to service calls.
	if (api_call == "service_create")
	  {
	    std::string sname = elts.at(1);
	    std::string body = elts.at(2);
	    calls_output.push_back(service_create(sname,body));
	  }
	else if (api_call == "service_predict")
	  {
	    std::string body = elts.at(1);
	    calls_output.push_back(service_predict(body));
	  }
	++lines;
      }
    _logger->info("Successfully executed calls from autostart JSON file {}",autostart_file);
    return dd_created_201();
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

  JDoc JsonAPI::dd_bad_request_400(const std::string &msg) const
  {
    JDoc jd;
    jd.SetObject();
    if (msg.empty())
      render_status(jd,400,"BadRequest");
    else render_status(jd,400,"BadRequest",400,msg);
    return jd;
  }
  
  JDoc JsonAPI::dd_forbidden_403() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,403,"Forbidden");
    return jd;
  }

  JDoc JsonAPI::dd_not_found_404() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"NotFound");
    return jd;
  }
  
  JDoc JsonAPI::dd_conflict_409() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,409,"Conflict");
    return jd;
  }

  JDoc JsonAPI::dd_internal_error_500(const std::string &msg) const
  {
    JDoc jd;
    jd.SetObject();
    if (msg.empty())
      render_status(jd,500,"InternalError");
    else render_status(jd,500,"InternalError",500,msg);
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

  JDoc JsonAPI::dd_service_input_bad_request_1005(const std::string &msg) const
  {
    JDoc jd;
    jd.SetObject();
    if (msg.empty())
      render_status(jd,400,"BadRequest",1005,"Service Input Error");
    else render_status(jd,400,"BadRequest",1005,"Service Input Error: " + msg);
    return jd;
  }
  
  JDoc JsonAPI::dd_service_bad_request_1006(const std::string &msg) const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,400,"BadRequest",1006,msg.empty() ? "Service Bad Request Error" : "Service Bad Request Error: " + msg);
    return jd;
  }
  
  JDoc JsonAPI::dd_internal_mllib_error_1007(const std::string &what) const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,500,"InternalError",1007,what);
    return jd;
  }

  JDoc JsonAPI::dd_train_predict_conflict_1008() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,409,"Conflict",1008,"Train / Predict Conflict");
    return jd;
  }

  JDoc JsonAPI::dd_output_connector_network_error_1009() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,404,"Not Found",1009,"Output Connector Network Error");
    return jd;
  }

#ifdef USE_SIMSEARCH
  JDoc JsonAPI::dd_sim_index_error_1010() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,403,"Forbidden",1010,"Cannot index after similarity search tree has been built");
    return jd;
  }

  JDoc JsonAPI::dd_sim_search_error_1011() const
  {
    JDoc jd;
    jd.SetObject();
    render_status(jd,403,"Forbidden",1011,"Cannot search before similarity search tree has been built");
    return jd;
  }
#endif
  
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

  JDoc JsonAPI::info(const std::string &jstr) const
  {
    bool status = false;

    if (!jstr.empty())
      {
	rapidjson::Document d;
	d.Parse(jstr.c_str());
	if (d.HasParseError())
	  {
	    _logger->error("JSON parsing error on string: {}",jstr);
	    return dd_bad_request_400();
	  }
	
	// parameters
	APIData ad;
	try
	  {
	    ad = APIData(d);
	    if (ad.has("status"))
	      status = ad.get("status").get<bool>();
	  }
	catch(RapidjsonException &e)
	  {
	    _logger->error("JSON error {}",e.what());
	    return dd_bad_request_400(e.what());
	  }
	catch(...)
	  {
	 return dd_bad_request_400();
	  }
      }
	
    // answer info call.
    JDoc jinfo = dd_ok_200();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/info",jinfo.GetAllocator());
    std::string sversion = std::to_string(VERSION_MAJOR) + "." + std::to_string(VERSION_MINOR);
    jhead.AddMember("version",JVal().SetString(sversion.c_str(),jinfo.GetAllocator()),jinfo.GetAllocator());
    jhead.AddMember("branch",JVal().SetString(GIT_BRANCH,jinfo.GetAllocator()),jinfo.GetAllocator());
    jhead.AddMember("commit",JVal().SetString(GIT_COMMIT_HASH,jinfo.GetAllocator()),jinfo.GetAllocator());
    JVal jservs(rapidjson::kArrayType);
    auto hit = _mlservices.begin();
    while(hit!=_mlservices.end())
      {
	APIData ad = mapbox::util::apply_visitor(visitor_info(status),(*hit).second);
	JVal jserv(rapidjson::kObjectType);
	ad.toJVal(jinfo,jserv);
	jservs.PushBack(jserv,jinfo.GetAllocator());
	++hit;
      }
    jhead.AddMember("services",jservs,jinfo.GetAllocator());
    jinfo.AddMember("head",jhead,jinfo.GetAllocator());
    return jinfo;
  }

  JDoc JsonAPI::service_create(const std::string &sname,
			       const std::string &jstr)
  {
    if (sname.empty())
      {
	_logger->error("missing service resource name: {}",sname);
	return dd_not_found_404();
      }

    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      {
	_logger->error("JSON parsing error on string: {}",jstr);
	return dd_bad_request_400();
      }

    std::string mllib,input;
    std::string type,description;
    bool store_config = false;
    APIData ad,ad_model;
    try
      {
	// mandatory parameters.
	mllib = d["mllib"].GetString();
	input = d["parameters"]["input"]["connector"].GetString();

	// optional parameters.
	if (d.HasMember("type"))
	  type = d["type"].GetString();
	else type = "supervised"; // default
	if (d.HasMember("description"))
	  description = d["description"].GetString();
	
	// model parameters (mandatory).
	ad = APIData(d);
	ad_model = ad.getobj("model");
	APIData ad_param = ad.getobj("parameters");
	if (ad_param.has("output"))
	  {
	    APIData ad_output = ad_param.getobj("output");
	    if (ad_output.has("store_config"))
	      store_config = ad_output.get("store_config").get<bool>();
	  }
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
        
    // create service.
    try
      {
	if (mllib.empty())
	  {
	    return dd_unknown_library_1000();
	  }
#ifdef USE_CAFFE
	else if (mllib == "caffe")
	  {
	    CaffeModel cmodel(ad_model,ad,_logger);
	    read_metrics_json(cmodel._repo,ad);
	    if (type == "supervised")
	      {
		if (input == "image")
		  add_service(sname,std::move(MLService<CaffeLib,ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "csv")
		  add_service(sname,std::move(MLService<CaffeLib,CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "csv_ts" || input == "csvts")
		  add_service(sname,std::move(MLService<CaffeLib,CSVTSCaffeInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "txt")
		  add_service(sname,std::move(MLService<CaffeLib,TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "svm")
		  add_service(sname,std::move(MLService<CaffeLib,SVMCaffeInputFileConn,SupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else return dd_input_connector_not_found_1004();
		if (JsonAPI::store_json_blob(cmodel._repo,jstr)) // store successful call json blob
		  _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,cmodel._repo);
		if (store_config)
		  if (JsonAPI::store_json_config_blob(cmodel._repo,jstr))
		    _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_config_blob_fname,cmodel._repo);
	      }
	    else if (type == "unsupervised")
	      {
		if (input == "image")
		  add_service(sname,std::move(MLService<CaffeLib,ImgCaffeInputFileConn,UnsupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "csv")
		  add_service(sname,std::move(MLService<CaffeLib,CSVCaffeInputFileConn,UnsupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "csv_ts")
		  add_service(sname,std::move(MLService<CaffeLib,CSVTSCaffeInputFileConn,UnsupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "txt")
		  add_service(sname,std::move(MLService<CaffeLib,TxtCaffeInputFileConn,UnsupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else if (input == "svm")
		  add_service(sname,std::move(MLService<CaffeLib,SVMCaffeInputFileConn,UnsupervisedOutput,CaffeModel>(sname,cmodel,description)),ad);
		else return dd_input_connector_not_found_1004();
		if (JsonAPI::store_json_blob(cmodel._repo,jstr)) // store successful call json blob
		  _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,cmodel._repo);
		if (store_config)
		  if (JsonAPI::store_json_config_blob(cmodel._repo,jstr))
		    _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_config_blob_fname,cmodel._repo);
	      }
	    else
	      {
		// unknown service type
		return dd_service_bad_request_1006();
	      }
	  }
#endif
#ifdef USE_CAFFE2

	else if (mllib == "caffe2") {
	  Caffe2Model c2model(ad_model,ad,_logger);
	  read_metrics_json(c2model._repo,ad);
	  if (type == "supervised") {

	    if (input == "image")
	      add_service(sname, std::move(MLService<Caffe2Lib, ImgCaffe2InputFileConn, SupervisedOutput, Caffe2Model>(sname, c2model, description)), ad);
	    else {
	      return dd_input_connector_not_found_1004();
	    }

	  } else if (type == "unsupervised") {

	    if (input == "image")
	      add_service(sname, std::move(MLService<Caffe2Lib, ImgCaffe2InputFileConn, UnsupervisedOutput, Caffe2Model>(sname, c2model, description)), ad);
	    else {
	      return dd_input_connector_not_found_1004();
	    }

	  } else return dd_service_bad_request_1006(); // unknown service type

	  // store successful call json blob
	  if (JsonAPI::store_json_blob(c2model._repo, jstr)) {
	    _logger->error("couldn't write {} file in model repository {}",
			   JsonAPI::_json_blob_fname, c2model._repo);
	  }

	  // store model configuration json blob
	  if (store_config && JsonAPI::store_json_config_blob(c2model._repo, jstr)) {
	    _logger->error("couldn't write {} file in model repository {}",
			   JsonAPI::_json_config_blob_fname, c2model._repo);
	  }
	}

#endif // USE_CAFFE2

#ifdef USE_NCNN
  else if (mllib == "ncnn")
  {
    NCNNModel ncnnmodel(ad_model,ad,_logger);
    read_metrics_json(ncnnmodel._repo,ad);
    if (type == "supervised")
    {
      if (input == "image")
        add_service(sname, std::move(MLService<NCNNLib,ImgNCNNInputFileConn,SupervisedOutput,NCNNModel>(sname,ncnnmodel,description)), ad);
      else if (input == "csv_ts" || input == "csvts")
        add_service(sname,std::move(MLService<NCNNLib,CSVTSNCNNInputFileConn,SupervisedOutput,NCNNModel>(sname,ncnnmodel,description)),ad);
      else return dd_input_connector_not_found_1004();
      if (JsonAPI::store_json_blob(ncnnmodel._repo, jstr))
        _logger->error("couldn't write {} file in model repository {}", JsonAPI::_json_blob_fname, ncnnmodel._repo);
      // store model configuration json blob
      if (store_config && JsonAPI::store_json_config_blob(ncnnmodel._repo, jstr)) {
	_logger->error("couldn't write {} file in model repository {}",
		       JsonAPI::_json_config_blob_fname, ncnnmodel._repo);
      }
    }
    else
      {
	// unknown type
	return dd_service_bad_request_1006();
      }
  }
#endif // USE_NCNN

#ifdef USE_TF
	else if (mllib == "tensorflow" || mllib == "tf")
	  {
	    TFModel tfmodel(ad_model,ad,_logger);
	    read_metrics_json(tfmodel._repo,ad);
	    if (type == "supervised")
	      {
		if (input == "image")
		  add_service(sname,std::move(MLService<TFLib,ImgTFInputFileConn,SupervisedOutput,TFModel>(sname,tfmodel,description)),ad);
		else return dd_input_connector_not_found_1004();
		if (JsonAPI::store_json_blob(tfmodel._repo,jstr)) // store successful call json blob
		  _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,tfmodel._repo);
	      }
	    else if (type == "unsupervised")
	      {
		if (input == "image")
		  add_service(sname,std::move(MLService<TFLib,ImgTFInputFileConn,UnsupervisedOutput,TFModel>(sname,tfmodel,description)),ad);
		else return dd_input_connector_not_found_1004();
		if (JsonAPI::store_json_blob(tfmodel._repo,jstr)) // store successful call json blob
		  _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,tfmodel._repo);
		if (store_config)
		  if (JsonAPI::store_json_config_blob(tfmodel._repo,jstr))
		    _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_config_blob_fname,tfmodel._repo);
	      }
	    else
	      {
		// unknown type
		return dd_service_bad_request_1006();
	      }
	  }
#endif
#ifdef USE_DLIB
    else if (mllib == "dlib") {
            DlibModel dlibmodel(ad_model,ad,_logger);
	    read_metrics_json(dlibmodel._repo,ad);
	    if (type == "supervised") {
	        if (input == "image") {
	            add_service(sname, std::move(MLService<DlibLib, ImgDlibInputFileConn, SupervisedOutput, DlibModel>(sname, dlibmodel, description)), ad);
            } else {
	            return dd_input_connector_not_found_1004();
            }
            if (JsonAPI::store_json_blob(dlibmodel._repo,jstr)) { // store successful call json blob
                _logger->error("couldn't write {} file in model repository {}", JsonAPI::_json_blob_fname, dlibmodel._repo);
            }
	    } else if (type == "unsupervised") {
	        return dd_service_bad_request_1006(); // Unsupervised not yet supported
	    } else {
	        // unknown type
	        return dd_service_bad_request_1006();
	    }
	}
#endif
#ifdef USE_XGBOOST
	else if (mllib == "xgboost")
	  {
	    XGBModel xmodel(ad_model,ad,_logger);
	    read_metrics_json(xmodel._repo,ad);
	    if (input == "csv")
	      add_service(sname,std::move(MLService<XGBLib,CSVXGBInputFileConn,SupervisedOutput,XGBModel>(sname,xmodel,description)),ad);
	    else if (input == "svm")
	      add_service(sname,std::move(MLService<XGBLib,SVMXGBInputFileConn,SupervisedOutput,XGBModel>(sname,xmodel,description)),ad);
	    else if (input == "txt")
	      add_service(sname,std::move(MLService<XGBLib,TxtXGBInputFileConn,SupervisedOutput,XGBModel>(sname,xmodel,description)),ad);
	    else return dd_input_connector_not_found_1004();
	    if (JsonAPI::store_json_blob(xmodel._repo,jstr)) // store successful call json blob
	      _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,xmodel._repo);
	    if (store_config)
	      if (JsonAPI::store_json_config_blob(xmodel._repo,jstr))
		_logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_config_blob_fname,xmodel._repo);
	  }
#endif
#ifdef USE_TSNE
	else if (mllib == "tsne")
	  {
	    TSNEModel tmodel(ad_model,ad,_logger);
	    read_metrics_json(tmodel._repo,ad);
	    if (input == "csv")
	      add_service(sname,std::move(MLService<TSNELib,CSVTSNEInputFileConn,UnsupervisedOutput,TSNEModel>(sname,tmodel,description)),ad);
	    else if (input == "txt")
	      add_service(sname,std::move(MLService<TSNELib,TxtTSNEInputFileConn,UnsupervisedOutput,TSNEModel>(sname,tmodel,description)),ad);
	    else return dd_input_connector_not_found_1004();
	    if (JsonAPI::store_json_blob(tmodel._repo,jstr)) // store successful call json blob
	      _logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_blob_fname,tmodel._repo);
	    if (store_config)
	      if (JsonAPI::store_json_config_blob(tmodel._repo,jstr))
		_logger->error("couldn't write {} file in model repository {}",JsonAPI::_json_config_blob_fname,tmodel._repo);
	  }
#endif
	else
	  {
	    return dd_unknown_library_1000();
	  }
      }
    catch (ServiceForbiddenException &e)
      {
	return dd_forbidden_403();
      }
    catch (InputConnectorBadParamException &e)
      {
	return dd_service_input_bad_request_1005(e.what());
      }
    catch (MLLibBadParamException &e)
      {
	return dd_service_bad_request_1006(e.what());
      }
    catch (InputConnectorInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (MLLibInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (std::exception &e)
      {
	return dd_internal_mllib_error_1007(e.what());
      }
    JDoc jsc = dd_created_201();
    return jsc;
  }
  
  JDoc JsonAPI::service_status(const std::string &sname)
  {
    if (sname.empty())
      return dd_service_not_found_1002();
    if (!this->service_exists(sname))
      return dd_not_found_404();
    auto hit = this->get_service_it(sname);
    APIData ad = mapbox::util::apply_visitor(visitor_status(),(*hit).second);
    JDoc jst = dd_ok_200();
    JVal jbody(rapidjson::kObjectType);
    ad.toJVal(jst,jbody);
    jst.AddMember("body",jbody,jst.GetAllocator());
    return jst;
  }

  JDoc JsonAPI::service_delete(const std::string &sname,
			       const std::string &jstr)
  {
    if (sname.empty())
      return dd_service_not_found_1002();
    
    rapidjson::Document d;
    if (!jstr.empty())
      {
	d.Parse(jstr.c_str());
	if (d.HasParseError())
	  {
	    _logger->error("JSON parsing error on string: {}",jstr);
	    return dd_bad_request_400();
	  }
      }

    APIData ad;
    try
      {
	if (!jstr.empty())
	  ad = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }

    try
      {
	if (remove_service(sname,ad))
	  return dd_ok_200();
      }
    catch (MLLibInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (std::exception &e)
      {
	return dd_internal_mllib_error_1007(e.what());
      }
    return dd_not_found_404();
  }

  JDoc JsonAPI::service_predict(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      {
	_logger->error("JSON parsing error on string: {}",jstr);
	return dd_bad_request_400();
      }

    // service
    std::string sname;
    try
      {
	sname = d["service"].GetString();
	std::transform(sname.begin(),sname.end(),sname.begin(),::tolower);
	if (!this->service_exists(sname))
	  return dd_service_not_found_1002();
      }
    catch(...)
      {
	return dd_bad_request_400();
      }

    // data
    APIData ad_data;
    try
      {
	ad_data = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    
    // prediction
    APIData out;
    try
      {
	this->predict(ad_data,sname,out); // we ignore returned status, stored in out data object
      }
    catch (InputConnectorBadParamException &e)
      {
	return dd_service_input_bad_request_1005(e.what());
      }
    catch (MLLibBadParamException &e)
      {
	return dd_service_bad_request_1006(e.what());
      }
    catch (InputConnectorInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (MLLibInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (MLServiceLockException &e)
      {
	return dd_train_predict_conflict_1008();
      }
#ifdef USE_SIMSEARCH
    catch (SimIndexException &e)
      {
	return dd_sim_index_error_1010();
      }
    catch (SimSearchException &e)
      {
	return dd_sim_search_error_1011();
      }
#endif
    catch (std::exception &e)
      {
	return dd_internal_mllib_error_1007(e.what());
      }
    JDoc jpred = dd_ok_200();
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jpred,jout);
    bool has_measure = ad_data.getobj("parameters").getobj("output").has("measure");
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/predict",jpred.GetAllocator());
    jhead.AddMember("service",d["service"],jpred.GetAllocator());
    if (!has_measure)
      jhead.AddMember("time",jout["time"],jpred.GetAllocator());
    jpred.AddMember("head",jhead,jpred.GetAllocator());
    if (has_measure)
      {
	jpred.AddMember("body",jout,jpred.GetAllocator());
	return jpred;
      }
    JVal jbody(rapidjson::kObjectType);
    if (jout.HasMember("predictions"))
      jbody.AddMember("predictions",jout["predictions"],jpred.GetAllocator());
    jpred.AddMember("body",jbody,jpred.GetAllocator());
    if (ad_data.getobj("parameters").getobj("output").has("template")
        && ad_data.getobj("parameters").getobj("output").get("template").get<std::string>() != "")
      {
	APIData ad_params = ad_data.getobj("parameters");
	APIData ad_output = ad_params.getobj("output");
	jpred.AddMember("template",JVal().SetString(ad_output.get("template").get<std::string>().c_str(),jpred.GetAllocator()),jpred.GetAllocator());
      }
    if (ad_data.getobj("parameters").getobj("output").has("network"))
      {
	APIData ad_params = ad_data.getobj("parameters");
	APIData ad_output = ad_params.getobj("output");
	APIData ad_net = ad_output.getobj("network");
	JVal jnet(rapidjson::kObjectType);
	ad_net.toJVal(jpred,jnet);
	jpred.AddMember("network",jnet,jpred.GetAllocator());
      }
    return jpred;
  }

  JDoc JsonAPI::service_train(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      {
	_logger->error("JSON parsing error on string: {}",jstr);
	return dd_bad_request_400();
      }
  
    // service
    std::string sname;
    try
      {
	sname = d["service"].GetString();
	std::transform(sname.begin(),sname.end(),sname.begin(),::tolower);
	if (!this->service_exists(sname))
	  return dd_service_not_found_1002();
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    
    // parameters and data
    APIData ad;
    try
      {
	ad = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    
    // training
    std::string mrepo;
    APIData out;
    try
      {	
	this->train(ad,sname,out); // we ignore return status, stored in out data object
	mrepo = out.getobj("model").get("repository").get<std::string>();
	if (JsonAPI::store_json_blob(mrepo,jstr)) // store successful call json blob
	  _logger->error("couldn't write to {} file in model repository {}",JsonAPI::_json_blob_fname,mrepo);
      }
    catch (InputConnectorBadParamException &e)
      {
	return dd_service_input_bad_request_1005(e.what());
      }
    catch (MLLibBadParamException &e)
      {
	return dd_service_bad_request_1006(e.what());
      }
    catch (InputConnectorInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (MLLibInternalException &e)
      {
	return dd_internal_error_500(e.what());
      }
    catch (std::exception &e)
      {
	return dd_internal_mllib_error_1007(e.what());
      }
    JDoc jtrain = dd_created_201();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/train",jtrain.GetAllocator());
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jtrain,jout);
    if (jout.HasMember("job")) // async job
      {
	jhead.AddMember("job",jout["job"].GetInt(),jtrain.GetAllocator());
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
    if (JsonAPI::store_json_blob(mrepo,jrender(jtrain))) // store successful call json blob
      _logger->error("couldn't write to {} file in model repository {}",JsonAPI::_json_blob_fname,mrepo);
    return jtrain;
  }

  JDoc JsonAPI::service_train_status(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      {
	_logger->error("JSON parsing error on string: {}",jstr);
	return dd_bad_request_400();
      }

    // service
    std::string sname;
    try
      {
	sname = d["service"].GetString();
	std::transform(sname.begin(),sname.end(),sname.begin(),::tolower);
	if (!this->service_exists(sname))
	  return dd_service_not_found_1002();
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    
    // parameters
    APIData ad;
    try
      {
	ad = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    if (!ad.has("job"))
      return dd_job_not_found_1003();
    
    // training status
    APIData out;
    int status = -2;
    JDoc dout; // this is to store any error message associated with the training job (e.g. if it has died)
    dout.SetObject();
    try
      {
	status = this->train_status(ad,sname,out);
      }
    catch (InputConnectorBadParamException &e)
      {
	dout = dd_service_input_bad_request_1005(e.what());
      }
    catch (MLLibBadParamException &e)
      {
	dout = dd_service_bad_request_1006(e.what());
      }
    catch (InputConnectorInternalException &e)
      {
	dout = dd_internal_error_500(e.what());
      }
    catch (MLLibInternalException &e)
      {
	dout = dd_internal_error_500(e.what());
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	dout = dd_bad_request_400(e.what());
      }
    catch (std::exception &e)
      {
	dout = dd_internal_mllib_error_1007(e.what());
      }
    JDoc jtrain;
    if (status == 1)
      {
	jtrain = dd_job_not_found_1003();
	JVal jhead(rapidjson::kObjectType);
	jhead.AddMember("method","/train",jtrain.GetAllocator());
	jhead.AddMember("job",ad.get("job").get<int>(),jtrain.GetAllocator());
	jtrain.AddMember("head",jhead,jtrain.GetAllocator());
	return jtrain;
      }
    jtrain = dd_ok_200();
    JVal jhead(rapidjson::kObjectType);
    jhead.AddMember("method","/train",jtrain.GetAllocator());
    jhead.AddMember("job",ad.get("job").get<int>(),jtrain.GetAllocator());
    JVal jout(rapidjson::kObjectType);
    std::string train_status;
    if (status != -2) // on failure, the output object from the async job is empty
      {
	out.toJVal(jtrain,jout);
	train_status = jout["status"].GetString();
	jhead.AddMember("status",JVal().SetString(jout["status"].GetString(),jtrain.GetAllocator()),jtrain.GetAllocator());
        jhead.AddMember("time",jout["time"].GetDouble(),jtrain.GetAllocator());
	jout.RemoveMember("time");
	jout.RemoveMember("status");
      }
    if (dout.HasMember("status"))
      {
	jhead.AddMember("status",JVal().SetString("error",jtrain.GetAllocator()),jtrain.GetAllocator());
	_logger->error(jrender(dout["status"]));
	JVal jvout(rapidjson::kObjectType);
	jvout.CopyFrom(dout["status"],jtrain.GetAllocator());
	jout.AddMember("Error",jvout,jtrain.GetAllocator());
      }
    jtrain.AddMember("head",jhead,jtrain.GetAllocator());
    jtrain.AddMember("body",jout,jtrain.GetAllocator());
    if (train_status == "finished" || train_status == "running")
      {
	std::string mrepo = out.getobj("model").get("repository").get<std::string>();
	if (JsonAPI::store_json_blob(mrepo,jrender(jtrain),"metrics.json"))
	  _logger->error("couldn't write to metrics.json file in model repository {}",mrepo);
      }
    return jtrain;
  }

  JDoc JsonAPI::service_train_delete(const std::string &jstr)
  {
    rapidjson::Document d;
    d.Parse(jstr.c_str());
    if (d.HasParseError())
      {
	_logger->error("JSON parsing error on string: {}",jstr);
	return dd_bad_request_400();
      }
  
    // service
    std::string sname;
    try
      {
	sname = d["service"].GetString();
	std::transform(sname.begin(),sname.end(),sname.begin(),::tolower);
	if (!this->service_exists(sname))
	  return dd_service_not_found_1002();
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    
    // parameters
    APIData ad;
    try
      {
	ad = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	_logger->error("JSON error {}",e.what());
	return dd_bad_request_400(e.what());
      }
    catch(...)
      {
	return dd_bad_request_400();
      }
    if (!ad.has("job"))
      return dd_job_not_found_1003();
    
    // delete training job
    APIData out;
    int status = this->train_delete(ad,sname,out);
    JDoc jd;
    if (status == 1)
      {
	jd = dd_job_not_found_1003();
	JVal jhead(rapidjson::kObjectType);
	jhead.AddMember("method","/train",jd.GetAllocator());
	jhead.AddMember("job",ad.get("job").get<int>(),jd.GetAllocator());
	jd.AddMember("head",jhead,jd.GetAllocator());
	return jd;
      }
    jd = dd_ok_200();
    JVal jout(rapidjson::kObjectType);
    out.toJVal(jd,jout);
    jout.AddMember("method","/train",jd.GetAllocator());
    jout.AddMember("job",ad.get("job").get<int>(),jd.GetAllocator());
    jd.AddMember("head",jout,jd.GetAllocator());
    return jd;
  }
  
  int JsonAPI::store_json_blob(const std::string &model_repo,
			       const std::string &jstr,
			       const std::string &jfilename)
  {
    std::ofstream outf;
    std::string outfname = model_repo + "/";
    if (!jfilename.empty())
      outfname += jfilename;
    else outfname += JsonAPI::_json_blob_fname;
    outf.open(outfname,std::ofstream::out|std::ofstream::trunc);
    if (!outf.is_open())
      return 1;
    outf << jstr << std::endl;
    return 0;
  }
  
  int JsonAPI::store_json_config_blob(const std::string &model_repo,
			       const std::string &jstr)
  {
    std::ofstream outf;
    outf.open(model_repo + "/" + JsonAPI::_json_config_blob_fname,std::ofstream::out|std::ofstream::trunc);
    if (!outf.is_open())
      return 1;
    outf << jstr << std::endl;
    return 0;
  }

  // read_json file blob to apidata
  int JsonAPI::read_json_blob(const std::string &model_repo,
			      const std::string &jfilename,
			      APIData &ad)
  {
    std::ifstream inf(model_repo + "/" + jfilename);
    std::stringstream buffer;
    try
      {
	buffer << inf.rdbuf();
    }
    catch (std::exception &e)
      {
	return 1;
      }
    rapidjson::Document d;
    d.Parse(buffer.str().c_str());
    if (d.HasParseError())
      {
	return 1;
      }
    try
      {
	ad = APIData(d);
      }
    catch(RapidjsonException &e)
      {
	return 1;
      }
    return 0;
  }

    // read_json file blob to apidata
  void JsonAPI::read_metrics_json(const std::string &model_repo,
				  APIData &ad)
  {
    APIData ad_metrics;
    if (read_json_blob(model_repo,"metrics.json",ad_metrics))
      {
	// quiet
      }
    else
      {
	APIData ad_metrics_body = ad_metrics.getobj("body");
	if (ad_metrics_body.has("measure_hist"))
	  {
	    APIData ad_metrics_hist = ad_metrics_body.getobj("measure_hist");
	    APIData ad_params = ad.getobj("parameters");
	    ad_params.add("metrics",ad_metrics_hist);
	    ad.add("parameters",ad_params);
	  }
      }
  }

}
