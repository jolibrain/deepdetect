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

#ifndef APIDATA_H
#define APIDATA_H

#include "utils/variant.hpp"
#include "ext/plustache/template.hpp"
//#include "ext/plustache/context.hpp"
#include <unordered_map>
#include <typeinfo>

namespace dd
{
  class APIData;
  
  //typedef mapbox::util::variant<std::string,int,double,bool> ad_variant_type;
  typedef mapbox::util::variant<std::string,double,bool,
    //mapbox::util::recursive_wrapper<APIData>> ad_variant_type;
    mapbox::util::recursive_wrapper<std::vector<APIData>>> ad_variant_type;

  class visitor_stache;
  
  class APIData
  {
  public:
    APIData() {}
    ~APIData() {}

    inline void add(const std::string &key, const ad_variant_type &val)
    {
      _data.insert(std::pair<std::string,ad_variant_type>(key,val));
    }

    inline ad_variant_type get(const std::string &key) const
    {
      std::unordered_map<std::string,ad_variant_type>::const_iterator hit;
      if ((hit=_data.find(key))!=_data.end())
	return (*hit).second;
      else return "";
    }

    //TODO: convert in and out from json.

    //TODO: convert out to custom template.
  public:
    inline std::string render_template(const std::string &tpl)
    {
      Plustache::Context ctx;
      to_plustache_ctx(ctx);
      Plustache::template_t t;
      return t.render(tpl,ctx);
    }

    void to_plustache_ctx(Plustache::Context &ctx) const;
    
    std::unordered_map<std::string,ad_variant_type> _data;
  };
  
  class visitor_stache : public mapbox::util::static_visitor<>
  {
  public:
    visitor_stache(Plustache::Context *ctx):_ctx(ctx) {}
    visitor_stache(PlustacheTypes::ObjectType *ot):_ot(ot) {}
    ~visitor_stache() {}
    
    void process(const std::string &str)
    {
      if (_ctx)
	_ctx->add(_key,str);
      else (*_ot)[_key] = str;
    }
    void process(const double &d)
    {
      if (_ctx)
	_ctx->add(_key,std::to_string(d));
      else (*_ot)[_key] = std::to_string(d);
    }
    void process(const bool &b)
    {
      if (_ctx)
	_ctx->add(_key,std::to_string(b));
      else (*_ot)[_key] = std::to_string(b);
    }
    void process(const std::vector<APIData> &vad)
    {
      if (_ctx) // only one level, as supported by mustache anyways.
	{
	  PlustacheTypes::CollectionType c;
	  for (size_t i=0;i<vad.size();i++)
	    {
	      PlustacheTypes::ObjectType ot;
	      visitor_stache vs(&ot);
	      APIData ad = vad.at(i);
	      auto hit = ad._data.begin();
	      while(hit!=ad._data.end())
		{
		  vs._key = (*hit).first;
		  mapbox::util::apply_visitor(vs,(*hit).second);
		  ++hit;
		}
	      c.push_back(ot);
	    }
	  _ctx->add(_key,c);
	}
    }

    template<typename T>
      void operator() (T &t)
      {
	process(t);
      }
    std::string _key;
    Plustache::Context *_ctx = nullptr;
    PlustacheTypes::ObjectType *_ot = nullptr;
  };
  
}

#endif
