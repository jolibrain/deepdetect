/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#ifndef CHAIN_H
#define CHAIN_H

#include "apidata.h"
#include <iostream>

namespace dd
{

  /**
   * \brief chain temporary data in between service calls
   */
  class ChainData
  {
  public:
    ChainData() {}
    ~ChainData() {}

    void add_model_data(const std::string &id,
			const APIData &out)
    {
      std::unordered_map<std::string,APIData>::iterator hit;
      if ((hit=_model_data.find(id))!=_model_data.end())
	_model_data.erase(hit);
      _model_data.insert(std::pair<std::string,APIData>(id,out));
    }

    APIData get_model_data(const std::string &id) const
    {
      std::unordered_map<std::string,APIData>::const_iterator hit;
      if ((hit=_model_data.find(id))!=_model_data.end())
	return (*hit).second;
      else
	return APIData();
    }

    void add_action_data(const std::string &id,
			 const APIData &out)
    {
      std::unordered_map<std::string,APIData>::iterator hit;
      if ((hit=_action_data.find(id))!=_action_data.end())
	_action_data.erase(hit);
      _action_data.insert(std::pair<std::string,APIData>(id,out));
    }

    APIData get_action_data(const std::string &id) const
    {
      std::unordered_map<std::string,APIData>::const_iterator hit;
      if ((hit=_action_data.find(id))!=_action_data.end())
	return (*hit).second;
      else
	return APIData();
    }

    void add_model_sname(const std::string &id,
			 const std::string &sname)
    {
      std::unordered_map<std::string,std::string>::iterator hit;
      if ((hit=_id_sname.find(id))==_id_sname.end())
	_id_sname.insert(std::pair<std::string,std::string>(id,sname));
    }

    std::string get_model_sname(const std::string &id)
    {
      std::unordered_map<std::string,std::string>::const_iterator hit;
      if ((hit=_id_sname.find(id))!=_id_sname.end())
	return (*hit).second;
      else return std::string();
    }
    
    APIData nested_chain_output();

    std::unordered_map<std::string,APIData> _model_data;
    std::unordered_map<std::string,APIData> _action_data;
    std::unordered_map<std::string,std::string> _id_sname;
    //std::string _first_sname;
    std::string _first_id;
  };

  /**
   * \brief building the chained model nested output
   */
  class visitor_nested
  {
  public:
    visitor_nested(std::unordered_map<std::string,APIData> *r)
      :_replacements(r) {}
    ~visitor_nested() {}
    
    void operator()(const std::string &str) { (void)str; }
    void operator()(const double &d) { (void)d; }
    void operator()(const int &i) { (void)i; }
    void operator()(const long int &i) { (void)i; }
    void operator()(const long long int &i) { (void)i; }
    void operator()(const bool &b) { (void)b; }
    void operator()(const std::vector<double> &vd) { (void)vd; }
    void operator()(const std::vector<int> &vd) { (void)vd; }
    void operator()(const std::vector<bool> &vd) { (void)vd; }
    void operator()(const std::vector<std::string> &vs) { (void)vs; }
    void operator()(const std::vector<cv::Mat> &vcv) { (void)vcv; }
    void operator()(const std::vector<std::pair<int,int>> &vpi) { (void)vpi; }
    void operator()(const APIData &ad);
    void operator()(const std::vector<APIData> &vad);
    
    std::unordered_map<std::string,APIData> *_replacements = nullptr;
    std::vector<APIData> _vad;
  };
  
}

#endif
