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

    double _loss = 0.0;
    std::map<double,std::string,std::greater<double>> _cats; /**< classes as finite set of categories. */
  };
  
}

#endif
