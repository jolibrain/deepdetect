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

#include "dto/predict_out.hpp"

namespace dd
{

  class UnsupervisedExtra
  {
  public:
    oatpp::Object<DTO::Dimensions> _imgsize;
    oatpp::UnorderedFields<DTO::DTOVector<double>> _confidences;
  };

  class UnsupervisedResult
  {
  public:
    UnsupervisedResult()
    {
    }

    UnsupervisedResult(const std::string &uri, const std::vector<double> &vals,
                       const UnsupervisedExtra &extra = UnsupervisedExtra(),
                       const std::string &meta_uri = "")
        : _uri(uri), _vals(vals), _extra(extra), _meta_uri(meta_uri)
    {
    }

    void binarized()
    {
      for (size_t i = 0; i < _vals.size(); i++)
        _vals.at(i) = _vals.at(i) <= 0.0 ? 0.0 : 1.0;
    }

    void bool_binarized()
    {
      for (size_t i = 0; i < _vals.size(); i++)
        _bvals.push_back(_vals.at(i) <= 0.0 ? false : true);
      _vals.clear();
    }

    void string_binarized()
    {
      for (size_t i = 0; i < _vals.size(); i++)
        _str += _vals.at(i) <= 0.0 ? "0" : "1";
      _vals.clear();
    }

#ifdef USE_SIMSEARCH
    void add_nn(const double dist, const std::string &uri)
    {
      _nns.insert(std::pair<double, std::string>(dist, uri));
    }
#endif

    std::string _uri;
    std::vector<double> _vals;
    std::vector<bool> _bvals;
    std::string _str;
    std::vector<cv::Mat> _images;
#ifdef USE_SIMSEARCH
    bool _indexed = false;
    std::multimap<double, std::string> _nns; /**< nearest neigbors. */
#endif
    // XXX(louis): this looks unused
    double _loss;
    // other metadata, e.g. image size.
    UnsupervisedExtra _extra;
    std::string _meta_uri; // used for indexing from a chain
  };

  /**
   * \brief supervised machine learning output connector class
   */
  class UnsupervisedOutput : public OutputConnectorStrategy
  {
  public:
    UnsupervisedOutput() : OutputConnectorStrategy()
    {
    }

    UnsupervisedOutput(const UnsupervisedOutput &uout)
        : OutputConnectorStrategy(uout)
    {
    }

    ~UnsupervisedOutput()
    {
    }

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

    void set_results(std::vector<UnsupervisedResult> &&results)
    {
      _vvres = std::move(results);
    }

    void add_results(const std::vector<APIData> &vrad)
    {
      std::unordered_map<std::string, int>::iterator hit;
      for (APIData ad : vrad)
        {
          std::string uri = ad.get("uri").get<std::string>();
          if (!ad.has("vals"))
            {
              this->_logger->error(
                  "unsupervised output needs mllib.extract_layer param");
              return;
            }

          std::vector<double> vals;
          if (!ad.get("vals").is<std::vector<cv::Mat>>())
            {
              vals = ad.get("vals").get<std::vector<double>>();
            }
          if ((hit = _vres.find(uri)) == _vres.end())
            {
              _vres.insert(std::pair<std::string, int>(uri, _vvres.size()));
              UnsupervisedExtra extra;

              if (ad.has("imgsize"))
                {
                  auto imgsize = ad.getobj("imgsize");
                  extra._imgsize = DTO::Dimensions::createShared();
                  extra._imgsize->width = imgsize.get("width").get<int>();
                  extra._imgsize->height = imgsize.get("height").get<int>();
                }

              if (ad.has("confidences"))
                {
                  extra._confidences = oatpp::UnorderedFields<
                      DTO::DTOVector<double>>::createShared();
                  auto confidences = ad.getobj("confidences");

                  for (std::string key : confidences.list_keys())
                    {
                      auto vec
                          = confidences.get(key).get<std::vector<double>>();
                      extra._confidences->emplace(std::make_pair(
                          key.c_str(),
                          DTO::DTOVector<double>(std::move(vec))));
                    }
                }
              std::string meta_uri;
              if (ad.has("index_uri"))
                meta_uri = ad.get("index_uri").get<std::string>();
              else if (ad.has("meta_uri"))
                meta_uri = ad.get("meta_uri").get<std::string>();
              _vvres.push_back(UnsupervisedResult(uri, vals, extra, meta_uri));
              if (ad.get("vals").is<std::vector<cv::Mat>>())
                {
                  _vvres.back()._images
                      = ad.get("vals").get<std::vector<cv::Mat>>();
                }
            }
        }
    }

    void finalize(const APIData &ad_in, APIData &ad_out, MLModel *mlm)
    {
      auto output_params = ad_in.createSharedDTO<DTO::OutputConnector>();
      finalize(output_params, ad_out, mlm);
    }

    void finalize(oatpp::Object<DTO::OutputConnector> output_params,
                  APIData &ad_out, MLModel *mlm)
    {
#ifndef USE_SIMSEARCH
      (void)mlm;
#endif
      _binarized = output_params->binarized;
      _bool_binarized = output_params->bool_binarized;
      _string_binarized = output_params->string_binarized;

      if (_binarized)
        {
          for (size_t i = 0; i < _vvres.size(); i++)
            {
              _vvres.at(i).binarized();
            }
        }
      else if (_bool_binarized)
        {
          for (size_t i = 0; i < _vvres.size(); i++)
            {
              _vvres.at(i).bool_binarized();
            }
        }
      else if (_string_binarized)
        {
          for (size_t i = 0; i < _vvres.size(); i++)
            {
              _vvres.at(i).string_binarized();
            }
        }

      std::unordered_set<std::string> indexed_uris;
#ifdef USE_SIMSEARCH
      if (output_params->index)
        {
          // check whether index has been created
          if (!mlm->_se)
            {
              int index_dim
                  = _vvres.at(0)._vals.size(); // XXX: lookup to the batch's
                                               // first output, as they should
                                               // all have the same size
              mlm->create_sim_search(index_dim, output_params);
            }

            // index output content -> vector (XXX: will need to flatten in
            // case of multiple vectors)
#ifdef USE_FAISS
          std::vector<URIData> urids;
          std::vector<std::vector<double>> vvals;
#endif
          for (size_t i = 0; i < _vvres.size(); i++)
            {
              URIData urid;
              if (_vvres.at(i)._meta_uri.empty())
                urid = URIData(_vvres.at(i)._uri);
              else
                urid = URIData(_vvres.at(i)._meta_uri);
#ifdef USE_FAISS
              urids.push_back(urid);
              vvals.push_back(_vvres.at(i)._vals);
#else
              mlm->_se->index(urid, _vvres.at(i)._vals);
#endif
              indexed_uris.insert(urid._uri);
            }
#ifdef USE_FAISS
          mlm->_se->index(urids, vvals);
#endif
        }
      if (output_params->build_index)
        {
          if (mlm->_se)
            mlm->build_index();
          else
            throw SimIndexException("Cannot build index if not created");
        }

      if (output_params->search)
        {
          if (!mlm->_se)
            {
              int index_dim
                  = _vvres.at(0)._vals.size(); // XXX: lookup to the batch's
                                               // first output, as they should
                                               // all have the same size
              mlm->create_sim_search(index_dim, output_params);
            }

          int search_nn = output_params->search_nn != nullptr
                              ? int(output_params->search_nn)
                              : _search_nn;
#ifdef USE_FAISS
          if (output_params->nprobe != nullptr)
            mlm->_se->_tse->_nprobe = output_params->nprobe;
#endif
          for (size_t i = 0; i < _vvres.size(); i++)
            {
              std::vector<URIData> nn_uris;
              std::vector<double> nn_distances;
              mlm->_se->search(_vvres.at(i)._vals, search_nn, nn_uris,
                               nn_distances);
              for (size_t j = 0; j < nn_uris.size(); j++)
                {
                  _vvres.at(i).add_nn(nn_distances.at(j), nn_uris.at(j)._uri);
                }
            }
        }
#endif

      to_ad(ad_out, indexed_uris);
    }

    void to_ad(APIData &out,
               const std::unordered_set<std::string> &indexed_uris) const
    {
#ifndef USE_SIMSEARCH
      (void)indexed_uris;
#endif
      auto out_dto = DTO::PredictBody::createShared();

      std::unordered_set<std::string>::const_iterator hit;
      for (size_t i = 0; i < _vvres.size(); i++)
        {
          auto pred_dto = DTO::Prediction::createShared();
          pred_dto->uri = _vvres.at(i)._uri.c_str();
          if (_vvres.at(i)._images.size() != 0)
            pred_dto->_images = _vvres.at(i)._images;
          if (_bool_binarized)
            pred_dto->vals
                = DTO::DTOVector<bool>(std::move(_vvres.at(i)._bvals));
          else if (_string_binarized)
            pred_dto->vals = oatpp::String(_vvres.at(i)._str.c_str());
          else
            pred_dto->vals
                = DTO::DTOVector<double>(std::move(_vvres.at(i)._vals));
          if (_vvres.at(i)._extra._imgsize)
            pred_dto->imgsize = _vvres.at(i)._extra._imgsize;
          if (_vvres.at(i)._extra._confidences != nullptr)
            pred_dto->confidences = _vvres.at(i)._extra._confidences;
          if (i == _vvres.size() - 1)
            pred_dto->last = true;
#ifdef USE_SIMSEARCH
          if (!indexed_uris.empty()
              && (hit = indexed_uris.find(_vvres.at(i)._uri))
                     != indexed_uris.end())
            pred_dto->indexed = true;
          if (!_vvres.at(i)._nns.empty())
            {
              pred_dto->nns = oatpp::Vector<oatpp::Any>::createShared();
              auto mit = _vvres.at(i)._nns.begin();
              while (mit != _vvres.at(i)._nns.end())
                {
                  auto nn = oatpp::UnorderedFields<oatpp::Any>::createShared();
                  nn->emplace("uri", oatpp::String((*mit).second.c_str()));
                  nn->emplace("dist", oatpp::Float64((*mit).first));
                  pred_dto->nns->push_back(nn);
                  ++mit;
                }
            }
#endif
          out_dto->predictions->push_back(pred_dto);
        }
      out.add("dto", out_dto);
    }

    std::unordered_map<std::string, int>
        _vres; /**< batch of results index, per uri. */
    std::vector<UnsupervisedResult> _vvres; /**< ordered results, per uri. */
    bool _binarized = false; /**< binary representation of output values. */
    bool _bool_binarized
        = false; /**< boolean binary representation of output values. */
    bool _string_binarized = false; /**< boolean string as binary
                                       representation of output values. */
#ifdef USE_SIMSEARCH
    int _search_nn = 10; /**< default nearest neighbors per search. */
#endif
  };
}

#endif
