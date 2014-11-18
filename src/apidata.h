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

namespace dd
{
  class APIData;
  
  /**
   * types:
   * 0: bool
   * 1: double
   * 2: int
   * 3: string
   */
  typedef mapbox::util::variant<std::string,int,double,bool> ad_variant_type;
    //mapbox::util::recursive_wrapper<APIData>> ad_variant_type;
    //mapbox::util::recursive_wrapper<std::vector<APIData>>> ad_variant_type;
  
  /*class visitor_stache : public mapbox::util::static_visitor<>
  {
  public:
    visitor_vad() {}
    ~visitor_vad() {}

    template<typename T>
      void operator() (T &t)
      {
	_ctx.add(_key,std::to_string(t));
      }
    std::string _key;
    Plustache::Context _ctx;
    };*/

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

    void to_plustache_ctx(Plustache::Context &ctx) const
    {
      //TODO: iterate _data and fill up context.
      //visitor_stache vs;
      auto hit = _data.begin();
      while(hit!=_data.end())
	{
	  //TODO: detect when nested type. get_type_index() ?
	  size_t vti = (*hit).second.get_type_index();
	  //std::cout << (*hit).first << " / " << vti << " / " << typeid((*hit).second).name() << std::endl;
	  //vs._key = (*hit).first;
	  //mapbox::util::apply_visitor(vs,(*hit).second);
	  if (vti == 0)
	    ctx.add((*hit).first,std::to_string((*hit).second.get<bool>()));
	  if (vti == 1)
	    ctx.add((*hit).first,std::to_string((*hit).second.get<double>()));
	  if (vti == 2)
	    ctx.add((*hit).first,std::to_string((*hit).second.get<int>()));
	  if (vti == 3)
	    ctx.add((*hit).first,(*hit).second.get<std::string>());
	  ++hit;
	}
    }
    
    std::unordered_map<std::string,ad_variant_type> _data;
  };
  
}

#endif
