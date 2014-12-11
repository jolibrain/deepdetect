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

#include "apidata.h"

namespace dd
{
  /*- visitor_vad -*/
  vout visitor_vad::process(const std::string &str)
  {
    return vout();
  }
  
  vout visitor_vad::process(const double &d)
  {
    return vout();
  }
  
  vout visitor_vad::process(const bool &b)
  {
    return vout();
  }
  
  vout visitor_vad::process(const std::vector<double> &vd)
  {
    return vout();
  }
  
  vout visitor_vad::process(const std::vector<std::string> &vs)
  {
    return vout();
  }
  
  vout visitor_vad::process(const std::vector<APIData> &vad)
  {
    return vout(vad);
  }
  
  template<typename T>
  vout visitor_vad::operator() (T &t)
  {
    return process(t);
  }

  /*- APIData -*/
  APIData::APIData(const JVal &jval)
  {
    fromJVal(jval);
  }

  void APIData::fromJVal(const JVal &jval)
  {
    for (rapidjson::Value::ConstMemberIterator cit=jval.MemberBegin();cit!=jval.MemberEnd();++cit)
      {
	if (cit->value.IsNull())
	  {
	  }
	else if (cit->value.IsBool())
	  {
	    add(cit->name.GetString(),cit->value.GetBool());
	  }
	else if (cit->value.IsObject())
	  {
	    APIData ad(jval[cit->name.GetString()]);
	    std::vector<APIData> vad = { ad };
	    add(cit->name.GetString(),vad);
	  }
	else if (cit->value.IsArray()) // only support arrays that bear a single type, number, string or object
	  {
	    const JVal &jarr = jval[cit->name.GetString()];
	    if (jarr.Size() != 0)
	      {
		if (jarr[0].IsNumber())
		  {
		    std::vector<double> vd;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      {
			if (jarr[i].IsDouble())
			  vd.push_back(jarr[i].GetDouble());
			else if (jarr[i].IsInt())
			  vd.push_back(static_cast<double>(jarr[i].GetInt()));
		      }
		    add(cit->name.GetString(),vd);
		  }
		else if (jarr[0].IsString())
		  {
		    std::vector<std::string> vs;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      vs.push_back(jarr[i].GetString());
		    add(cit->name.GetString(),vs);
		  }
		else if (jarr[0].IsObject())
		  {
		    std::vector<APIData> vad;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      {
			APIData nad;
			nad.fromJVal(jarr[i]);
			vad.push_back(nad);
		      }
		    add(cit->name.GetString(),vad);
		  }
	      }
	  }
	else if (cit->value.IsString())
	  {
	    add(cit->name.GetString(),cit->value.GetString());
	  }
	else if (cit->value.IsNumber())
	  {
	    add(cit->name.GetString(),cit->value.GetDouble());
	  }
      }
  }

  void APIData::toJDoc(JDoc &jd) const
  {
    visitor_rjson vrj(&jd);
    auto hit = _data.begin();
    while(hit!=_data.end())
      {
	vrj.set_key((*hit).first);
	mapbox::util::apply_visitor(vrj,(*hit).second);
	++hit;
      }
  }
  
  void APIData::toJVal(JDoc &jd, JVal &jv) const
  {
    visitor_rjson vrj(&jd,&jv);
    auto hit = _data.begin();
    while(hit!=_data.end())
      {
	vrj.set_key((*hit).first);
	mapbox::util::apply_visitor(vrj,(*hit).second);
	++hit;
      }
  }
  
  void APIData::to_plustache_ctx(Plustache::Context &ctx) const
  {
    visitor_stache vs(&ctx);
    auto hit = _data.begin();
    while(hit!=_data.end())
      {
	vs._key = (*hit).first;
	mapbox::util::apply_visitor(vs,(*hit).second);
	++hit;
      }
  }

  void APIData::to_plustache_ctx(Plustache::Context &ctx,
				 const std::string &key) const
  {
    auto hit = _data.find(key);
    if (hit!=_data.end())
      {
	visitor_stache vs(&ctx);
	vs._key = key;
	mapbox::util::apply_visitor(vs,(*hit).second);
      }
    else std::cout << "key not found when rendering template=" << key << std::endl;//TODO: else log error.
  }

}
