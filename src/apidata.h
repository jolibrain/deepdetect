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
#include "ext/rmustache/mustache.h"
#include "ext/rapidjson/rapidjson.h"
#include "ext/rapidjson/stringbuffer.h"
#include "ext/rapidjson/writer.h"
#include "dd_types.h"
#include <unordered_map>
#include <vector>
#include <sstream>
#include <typeinfo>

namespace dd
{
  class APIData;
  
  typedef mapbox::util::variant<std::string,double,bool,
    std::vector<std::string>,std::vector<double>,
    mapbox::util::recursive_wrapper<std::vector<APIData>>> ad_variant_type;

  class vout
  {
  public:
    vout() {}
    vout(const std::vector<APIData> &vad):_vad(vad) {}
    ~vout() {}
    std::vector<APIData> _vad;
  };
  class visitor_vad : public mapbox::util::static_visitor<vout>
  {
  public:
    visitor_vad() {}
    ~visitor_vad() {};
    
    vout process(const std::string &str);
    vout process(const double &d);
    vout process(const bool &b);
    vout process(const std::vector<double> &vd);
    vout process(const std::vector<std::string> &vs);
    vout process(const std::vector<APIData> &vad);
    
    template<typename T>
      vout operator() (T &t);
  };

  class APIData
  {
  public:
    APIData() {}
    APIData(const JVal &jval);
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
      else return ""; // beware
    }

    inline std::vector<APIData> getv(const std::string &key) const
    {
      visitor_vad vv;
      vout v = mapbox::util::apply_visitor(vv,get(key));
      return v._vad;
    }

    inline bool has(const std::string &key) const
    {
      std::unordered_map<std::string,ad_variant_type>::const_iterator hit;
      if ((hit=_data.find(key))!=_data.end())
	return true;
      else return false;
    }

    // convert in and out from json.
    void fromJVal(const JVal &jval);
    void toJDoc(JDoc &jd) const;
    void toJVal(JDoc &jd, JVal &jv) const;
    
  public:
    inline std::string render_template(const std::string &tpl)
    {
      std::stringstream ss;
      JDoc d;
      d.SetObject();
      toJDoc(d);

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      d.Accept(writer);
      std::string reststring = buffer.GetString();
      //std::cout << "to jdoc=" << reststring << std::endl;
      
      mustache::RenderTemplate(tpl, "", d, &ss);
      return ss.str();
    }
    
    std::unordered_map<std::string,ad_variant_type> _data;
  };

  class visitor_rjson : public mapbox::util::static_visitor<>
  {
  public:    
    visitor_rjson(JDoc *jd):_jd(jd) {}
    visitor_rjson(JDoc *jd, JVal *jv):_jd(jd),_jv(jv) {}
    visitor_rjson(const visitor_rjson &vrj):_jd(vrj._jd),_jv(vrj._jv)
      { _jvkey.CopyFrom(vrj._jvkey,_jd->GetAllocator()); }
    ~visitor_rjson() {}

    void set_key(const std::string &key)
    {
      _jvkey.SetString(key.c_str(),_jd->GetAllocator());
    }

    void process(const std::string &str)
    {
      if (!_jv)
	_jd->AddMember(_jvkey,JVal().SetString(str.c_str(),_jd->GetAllocator()),_jd->GetAllocator());
      else _jv->AddMember(_jvkey,JVal().SetString(str.c_str(),_jd->GetAllocator()),_jd->GetAllocator());
    }
    void process(const double &d)
    {
      if (!_jv)
	_jd->AddMember(_jvkey,JVal(d),_jd->GetAllocator());
      else _jv->AddMember(_jvkey,JVal(d),_jd->GetAllocator());
    }
    void process(const bool &b)
    {
      if (!_jv)
	_jd->AddMember(_jvkey,JVal(b),_jd->GetAllocator());
      else _jv->AddMember(_jvkey,JVal(b),_jd->GetAllocator());
    }
    void process(const std::vector<double> &vd)
    {
      JVal jarr(rapidjson::kArrayType);
      for (size_t i=0;i<vd.size();i++)
	{
	  jarr.PushBack(JVal(vd.at(i)),_jd->GetAllocator());
	}
      if (!_jv)
	_jd->AddMember(_jvkey,jarr,_jd->GetAllocator());
      else _jv->AddMember(_jvkey,jarr,_jd->GetAllocator());
    }
    void process(const std::vector<std::string> &vs)
    {
      JVal jarr(rapidjson::kArrayType);
      for (size_t i=0;i<vs.size();i++)
	{
	  jarr.PushBack(JVal().SetString(vs.at(i).c_str(),_jd->GetAllocator()),_jd->GetAllocator());
	}
      if (!_jv)
	_jd->AddMember(_jvkey,jarr,_jd->GetAllocator());
      else _jv->AddMember(_jvkey,jarr,_jd->GetAllocator());
    }
    void process(const std::vector<APIData> &vad)
    {
      JVal jov(rapidjson::kObjectType);
      if (vad.size() > 1)
	jov = JVal(rapidjson::kArrayType);
      for (size_t i=0;i<vad.size();i++)
	{
	  JVal jv(rapidjson::kObjectType); 
	  visitor_rjson vrj(_jd,&jv);
	  APIData ad = vad.at(i);
	  auto hit = ad._data.begin();
	  while(hit!=ad._data.end())
	    {
	      vrj.set_key((*hit).first);
	      mapbox::util::apply_visitor(vrj,(*hit).second);
	      ++hit;
	    }
	  if (vad.size() > 1)
	    jov.PushBack(jv,_jd->GetAllocator());
	  else jov = jv;
	}
      if (!_jv)
	_jd->AddMember(_jvkey,jov,_jd->GetAllocator());
      else _jv->AddMember(_jvkey,jov,_jd->GetAllocator());
    }

    template<typename T>
      void operator() (T &t)
      {
	process(t);
      }

    JVal _jvkey;
    JDoc *_jd = nullptr;
    JVal *_jv = nullptr;
  };
  
}

#endif
