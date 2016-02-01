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

  // recursive variant container, see utils/variant.hpp and utils/recursive_wrapper.hpp
  typedef mapbox::util::variant<std::string,double,int,bool,
    std::vector<std::string>,std::vector<double>,std::vector<int>,
    mapbox::util::recursive_wrapper<std::vector<APIData>>> ad_variant_type;

  /**
   * \brief data conversion exception
   */
  class DataConversionException : public std::exception
  {
  public:
    DataConversionException(const std::string &s)
      :_s(s) {}
    ~DataConversionException() {}
    const char *what() const noexcept { return _s.c_str(); }
  private:
    std::string _s;
  };
  
  /**
   * \brief object for visitor output
   */
  class vout
  {
  public:
    vout() {}
    vout(const std::vector<APIData> &vad):_vad(vad) {}
    ~vout() {}
    std::vector<APIData> _vad;
  };

  /**
   * \brief visitor class for easy access to variant vector container
   */
  class visitor_vad : public mapbox::util::static_visitor<vout>
  {
  public:
    visitor_vad() {}
    ~visitor_vad() {};
    
    vout process(const std::string &str);
    vout process(const double &d);
    vout process(const int &i);
    vout process(const bool &b);
    vout process(const std::vector<double> &vd);
    vout process(const std::vector<int> &vd);
    vout process(const std::vector<std::string> &vs);
    vout process(const std::vector<APIData> &vad);
    
    template<typename T>
      vout operator() (const T &t)
      {
	return process(t);
      }
  };

  /**
   * \brief main deepdetect API data object, uses recursive variant types
   */
  class APIData
  {
  public:
    /**
     * \brief empty constructor
     */
    APIData() {}
    
    /**
     * \brief constructor from rapidjson JSON object, see dd_types.h
     */
    APIData(const JVal &jval);

    APIData(const APIData &ad)
      :_data(ad._data)
    {}
    
    /**
     * \brief destructor
     */
    ~APIData() {}

    /**
     * \brief add key / object to data object
     * @param key string unique key
     * @param val variant value
     */
    inline void add(const std::string &key, const ad_variant_type &val)
    {
      auto hit = _data.begin();
      if ((hit=_data.find(key))!=_data.end())
	_data.erase(hit);
      _data.insert(std::pair<std::string,ad_variant_type>(key,val));
    }

    /**
     * \brief erase key / object from data object
     * @param key string unique key
     */
    inline void erase(const std::string &key)
    {
      auto hit = _data.begin();
      if ((hit=_data.find(key))!=_data.end())
	_data.erase(hit);
    }
    
    /**
     * \brief get value from data object
     *        at this stage, type of value is unknown and the typed object 
     *        must be later acquired with e.g. 'get<std::string>(val)
     * @param key string unique key
     * @return variant value
     */
    inline ad_variant_type get(const std::string &key) const
    {
      std::unordered_map<std::string,ad_variant_type>::const_iterator hit;
      if ((hit=_data.find(key))!=_data.end())
	return (*hit).second;
      else return ""; // beware
    }
    
    /**
     * \brief get vector container as variant value
     * @param key string unique value
     * @return vector of APIData as recursive variant value object
     */
    inline std::vector<APIData> getv(const std::string &key) const
    {
      visitor_vad vv;
      vout v = mapbox::util::apply_visitor(vv,get(key));
      return v._vad;
    }

    /**
     * \brief get data object value as variant value
     * @param key string unique value
     * @return APIData as recursive variant value object
     */
    inline APIData getobj(const std::string &key) const
    {
      visitor_vad vv;
      vout v = mapbox::util::apply_visitor(vv,get(key));
      if (v._vad.empty())
	return APIData();
      return v._vad.at(0);
    }

    /**
     * \brief find APIData object from vector, and that has a given key
     * @param vad vector of objects to search
     * @param key string unique key to look for
     * @return APIData as recursive variant value object
     */
    static APIData findv(const std::vector<APIData> &vad, const std::string &key)
    {
      for (const APIData &v : vad)
	{
	  if (v.has(key))
	    return v;
	}
      return APIData();
    }

    /**
     * \brief test whether the object contains a key at first level
     * @param key string unique key to look for
     * @return true if key is present, false otherwise
     */
    inline bool has(const std::string &key) const
    {
      std::unordered_map<std::string,ad_variant_type>::const_iterator hit;
      if ((hit=_data.find(key))!=_data.end())
	return true;
      else return false;
    }

    std::vector<std::string> list_keys() const
      {
	std::vector<std::string> keys;
	for (auto kv: _data)
	  {
	    keys.push_back(kv.first);
	  }
	return keys;
      }

    /**
     * \brief number of hosted keys at this level of the object
     * @return size
     */
    inline size_t size() const
    {
      return _data.size();
    }

    // convert in and out from json.
    /**
     * \brief converts rapidjson JSON to APIData
     * @param jval JSON object
     */
    void fromJVal(const JVal &jval);

    /**
     * \brief converts APIData to rapidjson JSON document
     * @param jd destination JSON Document
     */
    void toJDoc(JDoc &jd) const;

    /**
     * \brief converts APIData to rapidjson JSON value
     * @param jd JSON Document hosting the destination JSON value
     * @param jval destination JSON value
     */
    void toJVal(JDoc &jd, JVal &jv) const;
    
  public:
    /**
     * \brief render Mustache template based on this APIData object
     * @param tp template string
     */
    inline std::string render_template(const std::string &tpl)
    {
      std::stringstream ss;
      JDoc d;
      d.SetObject();
      toJDoc(d);

      /*rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      d.Accept(writer);
      std::string reststring = buffer.GetString();
      std::cout << "to jdoc=" << reststring << std::endl;*/
      
      mustache::RenderTemplate(tpl, "", d, &ss);
      return ss.str();
    }
    
    std::unordered_map<std::string,ad_variant_type> _data; /**< data as hashtable of variant types. */
  };

  /**
   * \brief visitor class for conversion to JSON
   */
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
    void process(const int &i)
    {
      if (!_jv)
	_jd->AddMember(_jvkey,JVal(i),_jd->GetAllocator());
      else _jv->AddMember(_jvkey,JVal(i),_jd->GetAllocator());
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
    void process(const std::vector<int> &vd)
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
