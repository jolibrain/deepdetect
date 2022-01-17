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

#include "dto/predict_out.hpp"

namespace dd
{

  void embed_model_output(
      oatpp::UnorderedFields<oatpp::Any> &dest,
      std::unordered_multimap<std::string, oatpp::Object<DTO::Prediction>>
          &other_models_out,
      const std::string &uri)
  {
    auto rhit_range = other_models_out.equal_range(uri);

    for (auto rhit = rhit_range.first; rhit != rhit_range.second; ++rhit)
      {
        oatpp::String model_name = rhit->second->uri;
        rhit->second->uri = nullptr;
        if (dest->find(model_name) != dest->end())
          throw ChainBadParamException(
              "This key already exists and cannot be used to reference "
              "model output: "
              + model_name->std_str());
        (*dest)[model_name] = rhit->second;
      }
  }

  oatpp::Object<DTO::ChainBody> ChainData::nested_chain_output()
  {
    // pre-compile models != first model
    std::vector<std::string> uris;
    std::shared_ptr<DTO::PredictBody> first_model_out;
    std::unordered_multimap<std::string, oatpp::Object<DTO::Prediction>>
        other_models_out;
    std::unordered_map<std::string, APIData>::const_iterator hit
        = _model_data.begin();
    while (hit != _model_data.end())
      {
        std::string model_id = (*hit).first;
        std::string model_name = get_model_sname(model_id);

        if (model_id == _first_id)
          {
            if ((*hit).second.has("dto"))
              {
                first_model_out
                    = hit->second.get("dto")
                          .get<oatpp::Any>()
                          .retrieve<oatpp::Object<DTO::PredictBody>>()
                          .getPtr();
              }
            else
              {
                // XXX: DTO conversion for supervised output
                first_model_out
                    = (*hit).second.createSharedDTO<DTO::PredictBody>();
              }
          }
        else
          {
            // predictions/classes or predictions/vals
            std::shared_ptr<DTO::PredictBody> body;
            if (hit->second.has("dto"))
              {
                body = hit->second.get("dto")
                           .get<oatpp::Any>()
                           .retrieve<oatpp::Object<DTO::PredictBody>>()
                           .getPtr();
              }
            else
              {
                // XXX: DTO conversion for supervised output
                body = hit->second.createSharedDTO<DTO::PredictBody>();
              }

            for (auto p : *body->predictions)
              {
                std::string uri = p->uri->std_str();
                p->uri = model_name.c_str();
                other_models_out.insert(
                    std::pair<std::string, oatpp::Object<DTO::Prediction>>(uri,
                                                                           p));
              }
          }
        ++hit;
      }

    // actions
    std::unordered_map<std::string, APIData>::const_iterator ahit
        = _action_data.begin();

    while (ahit != _action_data.end())
      {
        std::string action_id = ahit->first;
        const APIData &action_data = ahit->second;

        if (action_data.has("output"))
          {
            auto out_body = ahit->second.get("output")
                                .get<oatpp::Any>()
                                .retrieve<oatpp::Object<DTO::PredictBody>>();

            for (auto p : *out_body->predictions)
              {
                std::string uri = p->uri->std_str();
                p->uri = action_id.c_str();
                other_models_out.insert(
                    std::pair<std::string, oatpp::Object<DTO::Prediction>>(uri,
                                                                           p));
              }
          }
        ++ahit;
      }

    // Return a DTO
    auto chain_dto = DTO::ChainBody::createShared();
    for (auto pred : *first_model_out->predictions)
      {
        oatpp::UnorderedFields<oatpp::Any> chain_pred
            = oatpp_utils::dtoToUFields(pred);

        // chain result at uri level
        embed_model_output(chain_pred, other_models_out, pred->uri->std_str());

        // chain results at prediction level
        auto classes = oatpp::Vector<oatpp::Any>::createShared();

        for (auto cls : *pred->classes)
          {
            oatpp::UnorderedFields<oatpp::Any> class_preds
                = oatpp_utils::dtoToUFields(cls);

            if (cls->class_id != nullptr)
              {
                std::string uri = cls->class_id->std_str();
                cls->class_id = nullptr;
                embed_model_output(class_preds, other_models_out, uri);
              }
            classes->push_back(class_preds);
          }

        (*chain_pred)["classes"] = classes;
        chain_dto->predictions->push_back(chain_pred);
      }
    return chain_dto;
  }
}
