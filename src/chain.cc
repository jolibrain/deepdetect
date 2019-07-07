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

#include "chain.h"

namespace dd
{

  void visitor_nested::operator()(const APIData &ad)
  {
    (void)ad;
  }
  
  void visitor_nested::operator()(const std::vector<APIData> &vad)
  {
    //std::vector<std::string> kadd;
    std::unordered_map<std::string,APIData>::const_iterator rhit;
    for (size_t i=0;i<vad.size();i++)
      {
	APIData ad = vad.at(i);
	auto hit = ad._data.begin();
	while(hit!=ad._data.end())
	  {
	    //TODO: - if key is chainid -> replace + recursive visit to replacement
	    //      - else recursive visit
	    std::string ad_key = (*hit).first;
	    //std::cerr << "ad_key=" << ad_key << std::endl;
	    if ((rhit = _replacements->find(ad_key))!=_replacements->end()) 
	      {
		//std::cerr << "found key=" << ad_key << std::endl;

		//TODO: recursive replacements for chains with > 2 models
		//visitor_nested vn(_replacements);
		//mapbox::util::apply_visitor(vn,(*rhit).second);

		// this replaces the chainid object, not flexible for navigation as ids vary
		//(*hit).second = (*rhit).second; // replacement

		// we erase the chainid, and add up the model object
		ad._data.erase(hit);
		std::string nested_chain = (*rhit).second.list_keys().at(0);
		ad.add(nested_chain,
		       (*rhit).second.getobj(nested_chain));
		_vad.push_back(ad);
	      }
	    else
	      {
		APIData vis_ad_out;
		visitor_nested vn(_replacements);
		mapbox::util::apply_visitor(vn,(*hit).second);
		if (!vn._vad.empty())
		  {
		    vis_ad_out.add((*hit).first,vn._vad);
		    _vad.push_back(vis_ad_out);
		  }
	      }
	    ++hit;
	  }
      }
  }

  APIData ChainData::nested_chain_output()
  {
    //  pre-compile models != first model
    APIData first_model_out;
    std::unordered_map<std::string,APIData> other_models_out;
    std::unordered_map<std::string,APIData>::const_iterator hit = _model_data.begin();
    while(hit!=_model_data.end())
      {
	std::string model_name = (*hit).first;
	if (model_name == _first_sname)
	  first_model_out = (*hit).second;
	else
	  {
	    //TODO: predictions/classes or predictions/vals
	    std::vector<APIData> preds = (*hit).second.getv("predictions");
	    for (auto p: preds)
	      {
		if (p.has("classes"))
		  {
		    APIData clout;
		    APIData cls;
		    cls.add("classes",p.getv("classes"));
		    clout.add(model_name,cls);
		    other_models_out.insert(std::pair<std::string,APIData>(p.get("uri").get<std::string>(),
									   clout));
		  }
		else if (p.has("vals"))
		  {
		    APIData vout;
		    APIData vals;
		    vals.add("vals",p.get("vals").get<std::vector<double>>());
		    vout.add(model_name,vals);
		    other_models_out.insert(std::pair<std::string,APIData>(p.get("uri").get<std::string>(),
									   vout));
		  }
	      }
	  }
	++hit;
      }

    // call on nested visitor
    APIData vis_ad_out;
    visitor_nested vn(&other_models_out);
    auto vhit = first_model_out._data.begin();
    while(vhit!=first_model_out._data.end())
      {
	mapbox::util::apply_visitor(vn,(*vhit).second);
	if (!vn._vad.empty())
	  vis_ad_out.add((*vhit).first,vn._vad);
	++vhit;
      }
    return vis_ad_out;
  }

}
