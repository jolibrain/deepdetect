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
    void init(const APIData &ad);

    //TODO: output templating.
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
    class sup_result
    {
    public:
      sup_result(const double &loss=0.0)
	:_loss(loss) {}
      ~sup_result() {}
      inline void add_cat(const double &prob, const std::string &cat)
      {
	_cats.insert(std::pair<double,std::string>(prob,cat));
      }
      double _loss = 0.0;
      std::map<double,std::string,std::greater<double>> _cats;
    };

  public:
    SupervisedOutput()
      :OutputConnectorStrategy()
      {
      }
    SupervisedOutput(const SupervisedOutput &sout)
      :OutputConnectorStrategy(),_best(sout._best)
      {
      }
    ~SupervisedOutput() {}

    void init(const APIData &ad)
    {
      APIData ad_out = ad.getobj("parameters").getobj("output");
      if (ad_out.has("best"))
	_best = static_cast<int>(ad_out.get("best").get<double>());
    }

    inline void add_result(const std::string &uri, const double &loss)
    {
      std::unordered_map<std::string,sup_result>::iterator hit;
      if ((hit=_vcats.find(uri))==_vcats.end())
	_vcats.insert(std::pair<std::string,sup_result>(uri,sup_result(loss)));
      else (*hit).second._loss = loss;
    }

    inline void add_cat(const std::string &uri, const double &prob, const std::string &cat)
    {
      std::unordered_map<std::string,sup_result>::iterator hit;
      if ((hit=_vcats.find(uri))!=_vcats.end())
	(*hit).second.add_cat(prob,cat);
      // XXX: else error ?
    }

    void best_cats(const APIData &ad, SupervisedOutput &bcats) const
    {
      int best = _best;
      APIData ad_out = ad.getobj("parameters").getobj("output");
      if (ad_out.has("best"))
	best = static_cast<int>(ad_out.get("best").get<double>());
      auto hit = _vcats.begin();
      while(hit!=_vcats.end())
	{
	  sup_result bsresult((*hit).second._loss);
	  std::copy_n((*hit).second._cats.begin(),std::min(best,static_cast<int>((*hit).second._cats.size())),
		      std::inserter(bsresult._cats,bsresult._cats.end()));
	  bcats._vcats.insert(std::pair<std::string,sup_result>((*hit).first,bsresult));
	  ++hit;
	}
    }

    // for debugging purposes.
    void to_str(std::string &out) const
    {
      auto vit = _vcats.begin();
      while(vit!=_vcats.end())
	{
	  out += "-------------\n";
	  out += (*vit).first + "\n";
	  auto mit = (*vit).second._cats.begin();
	  while(mit!=(*vit).second._cats.end())
	  {
	    out += "accuracy=" + std::to_string((*mit).first) + " -- cat=" + (*mit).second + "\n";
	    ++mit;
	  }
	  ++vit;
	}
    }

    void to_ad(APIData &out) const
    {
      static std::string cl = "classes";
      static std::string phead = "prob";
      static std::string chead = "cat";
      std::vector<APIData> vpred;
      
      //debug
      /*std::string str;
      to_str(str);
      std::cout << "witness=\n" << str << std::endl;*/
      //debug
      
      auto hit = _vcats.begin();
      while(hit!=_vcats.end())
	{
	  int i = 0;
	  std::vector<APIData> v;
	  auto mit = (*hit).second._cats.begin();
	  while(mit!=(*hit).second._cats.end())
	    {
	      APIData nad;
	      nad.add(chead,(*mit).second);
	      nad.add(phead,(*mit).first);
	      v.push_back(nad);
	      ++i;
	      ++mit;
	    }
	  APIData adpred;
	  adpred.add(cl,v);
	  adpred.add("loss",(*hit).second._loss);
	  adpred.add("uri",(*hit).first);
	  vpred.push_back(adpred);
	  ++hit;
	}
      out.add("predictions",vpred);
    }
    
    std::unordered_map<std::string,sup_result> _vcats; /** batch of results, per uri. */
    
    // options
    int _best = 1;
  };
  
}

#endif
