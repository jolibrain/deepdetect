/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#ifndef UNSUPERVISEDOUTPUTCONNECTOR_H
#define UNSUPERVISEDOUTPUTCONNECTOR_H

namespace dd
{

  class unsup_result
  {
  public:
    unsup_result(const std::string &uri,
		 const std::vector<double> &vals)
      :_uri(uri),_vals(vals) {}

    ~unsup_result() {}

    std::string _uri;
    std::vector<double> _vals;
  };
  
  /**
   * \brief supervised machine learning output connector class
   */
  class UnsupervisedOutput : public OutputConnectorStrategy
  {
  public:
    UnsupervisedOutput()
      :OutputConnectorStrategy()
      {
      }

    UnsupervisedOutput(const UnsupervisedOutput &uout)
      :OutputConnectorStrategy()
      {
      }

    ~UnsupervisedOutput() {}

    void init(const APIData &ad)
    {
      //TODO: output connector parameters, if any
    }

    void add_results(const std::vector<APIData> &vrad)
    {
      //TODO
      std::unordered_map<std::string,int>::iterator hit;
      for (APIData ad: vrad)
	{
	  std::string uri = ad.get("uri").get<std::string>();
	  double loss = ad.get("loss").get<double>();
	  std::vector<double> vals = ad.get("vals").get<std::vector<double>>();
	  if ((hit=_vres.find(uri))==_vres.end())
	    {
	      _vres.insert(std::pair<std::string,int>(uri,_vvres.size()));
	      _vvres.push_back(unsup_result(uri,vals));
	    }
	}
    }

    void finalize(const APIData &ad_in, APIData &ad_out)
    {
      //TODO
      to_ad(ad_out);
    }

    void to_ad(APIData &out) const
    {
      std::vector<APIData> vpred;
      for (size_t i=0;i<_vvres.size();i++)
	{
	  APIData adpred;
	  adpred.add("uri",_vvres.at(i)._uri);
	  adpred.add("vals",_vvres.at(i)._vals);
	  if (i == _vvres.size()-1)
	    adpred.add("last",true);
	  vpred.push_back(adpred);
	}
      out.add("predictions",vpred);
    }
    
    std::unordered_map<std::string,int> _vres; /**< batch of results index, per uri. */
    std::vector<unsup_result> _vvres; /**< ordered results, per uri. */
  };

}

#endif
