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

#include "apidata.h"

namespace dd
{
  /*- visitor_vad -*/
  vout visitor_vad::operator()(const std::string &str)
  {
    (void)str;
    return vout();
  }
  
  vout visitor_vad::operator()(const double &d)
  {
    (void)d;
    return vout();
  }

  vout visitor_vad::operator()(const int &i)
  {
    (void)i;
    return vout();
  }

  vout visitor_vad::operator()(const long int &i)
  {
    (void)i;
    return vout();
  }
  
  vout visitor_vad::operator()(const bool &b)
  {
    (void)b;
    return vout();
  }

  vout visitor_vad::operator()(const APIData &ad)
  {
    return vout(ad);
  }
  
  vout visitor_vad::operator()(const std::vector<double> &vd)
  {
    (void)vd;
    return vout();
  }
  
  vout visitor_vad::operator()(const std::vector<int> &vd)
  {
    (void)vd;
    return vout();
  }
  
  vout visitor_vad::operator()(const std::vector<bool> &vd)
  {
    (void)vd;
    return vout();
  }

  vout visitor_vad::operator()(const std::vector<std::string> &vs)
  {
    (void)vs;
    return vout();
  }
  
  vout visitor_vad::operator()(const std::vector<APIData> &vad)
  {
    return vout(vad);
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
	else if (cit->value.IsArray()) // only supports array that bears a single type, number, string or object
	  {
	    const JVal &jarr = jval[cit->name.GetString()];
	    if (jarr.Size() != 0)
	      {
		if (jarr[0].IsDouble())
		  {
		    std::vector<double> vd;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      {
			vd.push_back(jarr[i].GetDouble());
		      }
		    add(cit->name.GetString(),vd);
		  }
		else if (jarr[0].IsInt())
		  {
		    std::vector<int> vd;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      {
			vd.push_back(jarr[i].GetInt());
		      }
		    add(cit->name.GetString(),vd);
		  }
		else if (jarr[0].IsBool())
		  {
		    std::vector<bool> vd;
		    for (rapidjson::SizeType i=0;i<jarr.Size();i++)
		      {
			vd.push_back(jarr[i].GetBool());
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
		else
		  {
		    throw DataConversionException("conversion error: unknown type of array");
		  }
	      }
	  }
	else if (cit->value.IsString())
	  {
	    add(cit->name.GetString(),std::string(cit->value.GetString()));
	  }
	else if (cit->value.IsDouble())
	  {
	    add(cit->name.GetString(),cit->value.GetDouble());
	  }
	else if (cit->value.IsInt())
	  {
	    add(cit->name.GetString(),cit->value.GetInt());
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
  
}
