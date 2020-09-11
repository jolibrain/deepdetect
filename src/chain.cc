/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#include "chain.h"

namespace dd
{

  void visitor_nested::operator()(const APIData &ad)
  {
    (void)ad;
  }

  void visitor_nested::operator()(const std::vector<APIData> &vad)
  {
    std::unordered_map<std::string, APIData>::iterator rhit;
    for (size_t i = 0; i < vad.size(); i++)
      {
        APIData ad = vad.at(i);
        APIData adc = ad;
        auto hit = ad._data.begin();
        while (hit != ad._data.end())
          {
            std::string ad_key = (*hit).first;
            auto rhit_range = _replacements->equal_range(ad_key);
            int rcount = std::distance(rhit_range.first, rhit_range.second);
            if (rcount > 0)
              {
                for (auto rhit = rhit_range.first; rhit != rhit_range.second;
                     ++rhit)
                  {
                    std::string nested_chain
                        = (*rhit).second.list_keys().at(0);

                    // recursive replacements for chains with > 2 models
                    bool recursive_changes = false;
                    visitor_nested vn(_replacements);
                    APIData nested_ad = (*rhit).second.getobj(nested_chain);
                    auto nhit = nested_ad._data.begin();
                    while (nhit != nested_ad._data.end())
                      {
                        mapbox::util::apply_visitor(vn, (*nhit).second);
                        if (!vn._vad.empty())
                          {
                            adc.add(nested_chain, vn._vad);
                            recursive_changes = true;
                          }
                        ++nhit;
                      }

                    if (!recursive_changes)
                      adc.add(nested_chain,
                              (*rhit).second.getobj(nested_chain));
                  }
                // we erase the chainid, and add up the model object
                adc._data.erase(ad_key);
                _vad.push_back(adc);
              }
            else
              {
                APIData vis_ad_out;
                visitor_nested vn(_replacements);
                mapbox::util::apply_visitor(vn, (*hit).second);
                if (!vn._vad.empty())
                  {
                    vis_ad_out.add((*hit).first, vn._vad);
                    _vad.push_back(vis_ad_out);
                  }
              }
            ++hit;
          }
      }
  }

  APIData ChainData::nested_chain_output()
  {
    //  pre-compile models != first model
    std::vector<std::string> uris;
    APIData first_model_out;
    std::unordered_multimap<std::string, APIData> other_models_out;
    std::unordered_map<std::string, APIData>::const_iterator hit
        = _model_data.begin();
    while (hit != _model_data.end())
      {
        std::string model_id = (*hit).first;
        std::string model_name = get_model_sname(model_id);
        if (model_id == _first_id)
          {
            first_model_out = (*hit).second;
            std::vector<APIData> predictions
                = first_model_out.getv("predictions");
            for (auto v : predictions)
              {
                uris.push_back(v.get("uri").get<std::string>());
              }
          }
        else
          {
            // predictions/classes or predictions/vals
            std::vector<APIData> preds = (*hit).second.getv("predictions");
            for (auto p : preds)
              {
                if (p.has("classes"))
                  {
                    APIData clout;
                    APIData cls;
                    cls.add("classes", p.getv("classes"));
                    clout.add(model_name, cls);
                    other_models_out.insert(std::pair<std::string, APIData>(
                        p.get("uri").get<std::string>(), clout));
                  }
                else if (p.has("vals"))
                  {
                    APIData vout;
                    APIData vals;
                    vals.add("vals", p.get("vals").get<std::vector<double>>());
                    if (p.has("nns"))
                      vals.add("nns", p.getv("nns"));
                    vout.add(model_name, vals);
                    other_models_out.insert(std::pair<std::string, APIData>(
                        p.get("uri").get<std::string>(), vout));
                  }
                else if (p.has("vector"))
                  {
                    APIData clout;
                    APIData cls;
                    cls.add("vector", p.getv("vector"));
                    clout.add(model_name, cls);
                    other_models_out.insert(std::pair<std::string, APIData>(
                        p.get("uri").get<std::string>(), clout));
                  }
              }
          }
        ++hit;
      }

    // call on nested visitor
    APIData vis_ad_out;
    visitor_nested vn(&other_models_out);
    auto vhit = first_model_out._data.begin();
    while (vhit != first_model_out._data.end())
      {
        mapbox::util::apply_visitor(vn, (*vhit).second);
        if (!vn._vad.empty())
          {
            vis_ad_out.add((*vhit).first, vn._vad);
          }
        ++vhit;
      }
    std::vector<APIData> predictions = vis_ad_out.getv("predictions");
    for (size_t i = 0; i < predictions.size(); i++)
      predictions.at(i).add("uri", uris.at(i));
    vis_ad_out.add("predictions", predictions);
    return vis_ad_out;
  }

}
