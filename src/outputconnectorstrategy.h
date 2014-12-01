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

#ifndef OUTPUTCONNECTORSTRATEGY_H
#define OUTPUTCONNECTORSTRATEGY_H

#include <map>
#include <iostream>

namespace dd
{

  class OutputConnectorStrategy
  {
  public:
    OutputConnectorStrategy() {}
    ~OutputConnectorStrategy() {}

    int transform() { return 1; }
  };

  class NoOutputConn : public OutputConnectorStrategy
  {
  public:
    NoOutputConn()
      :OutputConnectorStrategy() {}
    ~NoOutputConn() {}
  };

  class SupervisedOutput : public OutputConnectorStrategy
  {
  public:
    SupervisedOutput()
      :OutputConnectorStrategy()
      {}
    ~SupervisedOutput() {}

    inline void add_cat(const double &prob, const std::string &cat)
    {
      _cats.insert(std::pair<double,std::string>(prob,cat));
    }

    void best_cats(const int &num, SupervisedOutput &bcats) const
    {
      std::copy_n(_cats.begin(),std::min(num,static_cast<int>(_cats.size())),std::inserter(bcats._cats,bcats._cats.end()));
    }

    //TODO: e.g. to_json, print, ...
    void to_str(std::string &out) const
    {
	auto mit = _cats.begin();
	while(mit!=_cats.end())
	  {
	    out += "accuracy=" + std::to_string((*mit).first) + " -- cat=" + (*mit).second + "\n";
	    ++mit;
	  }
    }

    void to_ad(APIData &out) const
    {
      static std::string cl = "classes";
      static std::string phead = "prob";
      static std::string chead = "cat";
      std::vector<APIData> v;
     
      //debug
      /*std::string str;
      to_str(str);
      std::cout << "witness=\n" << str << std::endl;*/
      //debug
      
      int i = 0;
      auto mit = _cats.begin();
      while(mit!=_cats.end())
	{
	  APIData nad;
	  nad.add(chead,(*mit).second);
	  nad.add(phead,(*mit).first);
	  v.push_back(nad);
	  /*std::vector<std::pair<std::string,std::string>> v;
	  v.push_back(std::pair<std::string,std::string>(cchead,(*mit).second));
	  v.push_back(std::pair<std::string,std::string>(phead,std::to_string((*mit).first)));
	  ad_classes.*/
	  
	  /*std::string s = std::to_string(i);
	  out.add(chead + s,(*mit).second);
	  out.add(phead + s,static_cast<double>((*mit).first));*/
	  ++i;
	  ++mit;
	}
      //out.add(cl,ad_classes);
      APIData adpred;
      adpred.add(cl,v);
      adpred.add("loss",_loss);
      std::vector<APIData> vpred = { adpred };
      out.add("predictions",vpred);
      /*out.add("bool",true);
      out.add("string","bla");
      out.add("int",3);
      out.add("double",8.0);*/
    }

    double _loss = 0.0;
    std::map<double,std::string,std::greater<double>> _cats; /**< classes as finite set of categories. */
  };
  
}

#endif
