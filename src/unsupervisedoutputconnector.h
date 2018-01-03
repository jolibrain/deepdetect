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

    void binarized()
    {
      for (size_t i=0;i<_vals.size();i++)
	_vals.at(i) = _vals.at(i) <= 0.0 ? 0.0 : 1.0;
    }
    
    void bool_binarized()
    {
      for (size_t i=0;i<_vals.size();i++)
	_bvals.push_back(_vals.at(i) <= 0.0 ? false : true);
      _vals.clear();
    }

    void string_binarized()
    {
       for (size_t i=0;i<_vals.size();i++)
	_str += _vals.at(i) <= 0.0 ? "0" : "1";
      _vals.clear();
    }

    void add_nn(const double dist, const std::string &uri)
    {
      _nns.insert(std::pair<double,std::string>(dist,uri));
    }
    
    std::string _uri;
    std::vector<double> _vals;
    std::vector<bool> _bvals;
    std::string _str;
    bool _indexed = false;
    std::multimap<double,std::string> _nns; /**< nearest neigbors. */
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
	(void)uout;
      }

    ~UnsupervisedOutput() {}

    void init(const APIData &ad)
    {
      APIData ad_out = ad.getobj("parameters").getobj("output");
      if (ad_out.has("binarized"))
	_binarized = ad_out.get("binarized").get<bool>();
      else if (ad_out.has("bool_binarized"))
	_bool_binarized = ad_out.get("bool_binarized").get<bool>();
      else if (ad_out.has("string_binarized"))
	_string_binarized = ad_out.get("string_binarized").get<bool>();
    }
    
    void add_results(const std::vector<APIData> &vrad)
    {
      std::unordered_map<std::string,int>::iterator hit;
      for (APIData ad: vrad)
	{
	  std::string uri = ad.get("uri").get<std::string>();
	  //double loss = ad.get("loss").get<double>();
	  std::vector<double> vals = ad.get("vals").get<std::vector<double>>();
	  if ((hit=_vres.find(uri))==_vres.end())
	    {
	      _vres.insert(std::pair<std::string,int>(uri,_vvres.size()));
	      _vvres.push_back(unsup_result(uri,vals));
	    }
	}
    }

    void finalize(const APIData &ad_in, APIData &ad_out, MLModel *mlm)
    {
      if (ad_in.has("binarized"))
	_binarized = ad_in.get("binarized").get<bool>();
      else if (ad_in.has("bool_binarized"))
	_bool_binarized = ad_in.get("bool_binarized").get<bool>();
      else if (ad_in.has("string_binarized"))
	_string_binarized = ad_in.get("string_binarized").get<bool>();
      if (_binarized)
	{
	  for (size_t i=0;i<_vvres.size();i++)
	    {
	      _vvres.at(i).binarized();
	    }
	}
      else if (_bool_binarized)
	{
	  for (size_t i=0;i<_vvres.size();i++)
	    {
	      _vvres.at(i).bool_binarized();
	    }
	}
      else if (_string_binarized)
	{
	  for (size_t i=0;i<_vvres.size();i++)
	    {
	      _vvres.at(i).string_binarized();
	    }
	}

      std::unordered_set<std::string> indexed_uris;
      if (ad_in.has("index") && ad_in.get("index").get<bool>())
	{
	  // check whether index has been created
	  if (!mlm->_se)
	    {
	      int index_dim = _vvres.at(0)._vals.size(); //XXX: lookup to the batch's first output, as they should all have the same size
	      std::cerr << "Creating index\n";
	      mlm->create_sim_search(index_dim);
	    }
	      
	  // index output content -> vector (XXX: will need to flatten in case of multiple vectors)
	  for (size_t i=0;i<_vvres.size();i++)
	    {
	      mlm->_se->index(_vvres.at(i)._uri,_vvres.at(i)._vals);
	      indexed_uris.insert(_vvres.at(i)._uri);
	    }
	}
      if (ad_in.has("build_index") && ad_in.get("build_index").get<bool>())
	{
	  if (mlm->_se)
	    mlm->build_index();
	  else throw SimIndexException("Cannot build index if not created");
	}

      if (ad_in.has("search") && ad_in.get("search").get<bool>())
	{
	  int search_nn = _search_nn;
	  if (ad_in.has("search_nn"))
	    search_nn = ad_in.get("search_nn").get<int>();
	  for (size_t i=0;i<_vvres.size();i++)
	    {
	      std::vector<std::string> nn_uris;
	      std::vector<double> nn_distances;
	      mlm->_se->search(_vvres.at(i)._vals,search_nn,nn_uris,nn_distances);
	      for (size_t j=0;j<nn_uris.size();j++)
		{
		  _vvres.at(i).add_nn(nn_distances.at(i),nn_uris.at(i));
		}
	    }
	}
      
      to_ad(ad_out,indexed_uris);
    }

    void to_ad(APIData &out, const std::unordered_set<std::string> &indexed_uris) const
    {
      std::unordered_set<std::string>::const_iterator hit;
      std::vector<APIData> vpred;
      for (size_t i=0;i<_vvres.size();i++)
	{
	  APIData adpred;
	  adpred.add("uri",_vvres.at(i)._uri);
	  if (_bool_binarized)
	    adpred.add("vals",_vvres.at(i)._bvals);
	  else if (_string_binarized)
	    adpred.add("vals",_vvres.at(i)._str);
	  else adpred.add("vals",_vvres.at(i)._vals);
	  if (i == _vvres.size()-1)
	    adpred.add("last",true);
	  if (!indexed_uris.empty() && (hit=indexed_uris.find(_vvres.at(i)._uri))!=indexed_uris.end())
	    adpred.add("indexed",true);
	  if (!_vvres.at(i)._nns.empty())
	    {
	      std::vector<APIData> ad_nns;
	      auto mit = _vvres.at(i)._nns.begin();
	      while(mit!=_vvres.at(i)._nns.end())
		{
		  APIData ad_nn;
		  ad_nn.add("uri",(*mit).second);
		  ad_nn.add("dist",(*mit).first);
		  ad_nns.push_back(ad_nn);
		  ++mit;
		}
	      adpred.add("nns",ad_nns);
	    }
	  vpred.push_back(adpred);
	}
      out.add("predictions",vpred);
    }
    
    std::unordered_map<std::string,int> _vres; /**< batch of results index, per uri. */
    std::vector<unsup_result> _vvres; /**< ordered results, per uri. */
    bool _binarized = false; /**< binary representation of output values. */
    bool _bool_binarized = false; /**< boolean binary representation of output values. */
    bool _string_binarized = false; /**< boolean string as binary representation of output values. */
    int _search_nn = 10; /**< default nearest neighbors per search. */
  };

}

#endif
