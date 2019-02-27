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

#ifndef JSONAPI_H
#define JSONAPI_H

#include "apistrategy.h"
#include "dd_types.h"

namespace dd
{
  /**
   * \brief JSON API class
   */
  class JsonAPI : public APIStrategy
  {
  public:
    JsonAPI();
    ~JsonAPI();

    /**
     * \brief command line handling
     */
    int boot(int argc, char *argv[]);

    /**
     * \brief service autostart from JSON file
     * @param autostart_file JSON file with service API call and JSON body
     */
    JDoc service_autostart(const std::string &autostart_file);
    
    /**
     * \brief error status generation
     * @param jst JSON document object
     * @param code error HTTP code
     * @param msg error message
     * @param dd_code deepdetect custom error code
     * @param dd_msg deepdetect custom error message
     */
    void render_status(JDoc &jst,
		       const uint32_t &code, const std::string &msg,
		       const uint32_t &dd_code=0, const std::string &dd_msg="") const;
    
    // errors
    JDoc dd_ok_200() const;
    JDoc dd_created_201() const;
    JDoc dd_bad_request_400(const std::string &msg="") const;
    JDoc dd_forbidden_403() const;
    JDoc dd_not_found_404() const;
    JDoc dd_conflict_409() const;
    JDoc dd_internal_error_500(const std::string &msg="") const;

    // specific errors
    JDoc dd_unknown_library_1000() const;
    JDoc dd_no_data_1001() const;
    JDoc dd_service_not_found_1002() const;
    JDoc dd_job_not_found_1003() const;
    JDoc dd_input_connector_not_found_1004() const;
    JDoc dd_service_input_bad_request_1005(const std::string &what="") const;
    JDoc dd_service_bad_request_1006(const std::string &what="") const;
    JDoc dd_internal_mllib_error_1007(const std::string &what) const;
    JDoc dd_train_predict_conflict_1008() const;
    JDoc dd_output_connector_network_error_1009() const;
    JDoc dd_sim_index_error_1010() const;
    JDoc dd_sim_search_error_1011() const;
    
    // JSON rendering
    std::string jrender(const JDoc &jst) const;
    std::string jrender(const JVal &jval) const;

    // resources
    // return a JSON document for every API call
    JDoc info(const std::string &jstr) const;
    JDoc service_create(const std::string &sname, const std::string &jstr);
    JDoc service_status(const std::string &sname);
    JDoc service_delete(const std::string &sname,
			const std::string &jstr);
    
    JDoc service_predict(const std::string &jstr);

    JDoc service_train(const std::string &jstr);
    JDoc service_train_status(const std::string &jstr);
    JDoc service_train_delete(const std::string &jstr);

    static int store_json_blob(const std::string &model_repo,
			       const std::string &jstr,
			       const std::string &jfilename="");

    static int store_json_config_blob(const std::string &model_repo,
				      const std::string &jstr);

    static int read_json_blob(const std::string &model_repo,
			      const std::string &jfilename,
			      APIData &ad);

    static void read_metrics_json(const std::string &model_repo,
				  APIData &ad);
    
    static std::string _json_blob_fname;
    static std::string _json_config_blob_fname;
    //std::string _mrepo; /**< service file repository */
  };

  /**
   * \brief visitor class for service info call
   */
  class visitor_info : public mapbox::util::static_visitor<APIData>
  {
  public:
    visitor_info(const bool &status):_status(status) {}
    ~visitor_info() {}

    template<typename T>
      APIData operator() (T &mllib)
      {
	return mllib.info(_status);
      }
    bool _status = false;
  };

  /**
   * \brief visitor class for service status call
   */
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
