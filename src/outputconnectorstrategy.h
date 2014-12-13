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
      {
      }
    ~SupervisedOutput() {}

    inline void add_cat(const int &pos, const double &prob, const std::string &cat, const double &loss=0.0)
    {
      if (pos >= static_cast<int>(_vcats.size()))
	{
	  _vcats.resize(pos+1);
	  _losses.resize(pos+1);
	}
      _vcats.at(pos).insert(std::pair<double,std::string>(prob,cat));
      _losses.at(pos) = loss;
    }

    void best_cats(const int &num, SupervisedOutput &bcats) const
    {
      bcats._vcats.resize(_vcats.size());
      for (size_t i=0;i<_vcats.size();i++)
	std::copy_n(_vcats.at(i).begin(),std::min(num,static_cast<int>(_vcats.at(i).size())),std::inserter(bcats._vcats.at(i),bcats._vcats.at(i).end()));
      bcats._losses = _losses;
    }

    // for debugging purposes.
    void to_str(std::string &out) const
    {
      auto vit = _vcats.begin();
      while(vit!=_vcats.end())
	{
	  out += "-------------\n";
	  auto mit = (*vit).begin();
	  while(mit!=(*vit).end())
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
      
      for (size_t j=0;j<_vcats.size();j++)
	{
	  int i = 0;
	  std::vector<APIData> v;
	  auto mit = _vcats.at(j).begin();
	  while(mit!=_vcats.at(j).end())
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
	  adpred.add("loss",_losses.at(j));
	  vpred.push_back(adpred);
	}
      out.add("predictions",vpred);
    }

    std::vector<double> _losses;
    std::vector<std::map<double,std::string,std::greater<double>>> _vcats; /**< vector of classes as finite set of categories. */
  };
  
}

#endif
