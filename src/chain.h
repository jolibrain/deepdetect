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

#include "chain_actions.h"
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

    void add_model_data(const std::string &sname,
			const APIData &out)
    {
      auto hit = _model_data.begin();
      if ((hit=_model_data.find(sname))!=_model_data.end())
	_model_data.erase(hit);
      _model_data.insert(std::pair<std::string,APIData>(sname,out));
    }

    APIData get_model_data(const std::string &sname) const
    {
      std::unordered_map<std::string,APIData>::const_iterator hit;
      if ((hit = _model_data.find(sname))!=_model_data.end())
	return (*hit).second;
      else
	{
	  std::cerr << "[chain] could not find model data for service " << sname << std::endl;
	  return APIData();
	}
    }

    /*void add_action_data(const std::string &aname,
			 const APIData &out)
    {
      _action_data.insert(std::pair<std::string,APIData>(aname,out));
      }*/

    /*APIData get_action_data(const std::string &aname) const
    {
      std::unordered_map<std::string,APIData>::const_iterator hit;
      if ((hit = _action_data.find(aname))!=_action_data.end())
	return (*hit).second;
      else
	{
	  std::cerr << "[chain] could not find action data for action " << aname << std::endl;
	  return APIData();
	}
	}*/

    APIData nested_chain_output();

    std::unordered_map<std::string,APIData> _model_data;
    std::vector<APIData> _action_data;
    std::string _first_sname;
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
    std::string _key;
    APIData _ad;
    std::vector<APIData> _vad;
  };
  
}

#endif
