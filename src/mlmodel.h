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

#ifndef MODEL_H
#define MODEL_H

#ifdef USE_SIMSEARCH
#include "simsearch.h"
#endif
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "apidata.h"
#include "dto/model.hpp"
#include "utils/fileops.hpp"
#ifndef WIN32
#include "utils/httpclient.hpp"
#endif
#include "mllibstrategy.h"
#include "dto/service_create.hpp"
#include "dto/parameters.hpp"

namespace dd
{
  class MLModel
  {
  public:
    MLModel()
    {
    }

    MLModel(const oatpp::Object<DTO::Model> &model_dto,
            const oatpp::Object<DTO::ServiceCreate> &service_dto,
            const std::shared_ptr<spdlog::logger> &logger)
    {
      init_repo_dir(model_dto, logger.get());
      if (model_dto->init)
        read_config_json(service_dto);
    }

    MLModel(const oatpp::Object<DTO::Model> &model_dto)
    {
      init_repo_dir(model_dto, nullptr);
    }

    MLModel(const std::string &repo) : _repo(repo)
    {
    }

    MLModel(const oatpp::Object<DTO::Model> &model_dto, const std::string &repo) : _repo(repo)
    {
      init_repo_dir(model_dto, nullptr);
    }

    ~MLModel()
    {
#ifdef USE_SIMSEARCH
      delete _se;
#endif
    }

    void read_corresp_file()
    {
      if (!_corresp.empty())
        {
          std::ifstream ff(_corresp);
          if (!ff.is_open())
            std::cerr << "cannot open model corresp file=" << _corresp
                      << std::endl;
          else
            {
              std::string line;
              while (!ff.eof())
                {
                  std::getline(ff, line);
                  std::string key = line.substr(0, line.find(' '));
                  if (!key.empty())
                    {
                      std::string value = line.substr(line.find(' ') + 1);
                      _hcorresp.insert(
                          std::pair<int, std::string>(std::stoi(key), value));
                    }
                }
            }
        }
    }

    inline std::string get_hcorresp(const int &i)
    {
      if (_hcorresp.empty())
        return std::to_string(i);
      else
        return _hcorresp[i];
    }

#ifdef USE_SIMSEARCH
    /**
     * \brief create similarity search engine
     */
    void create_sim_search(const int &dim, const APIData ad)
    {
      if (!_se)
        {
#ifdef USE_ANNOY
          (void)ad;
          _se = new SearchEngine<AnnoySE>(dim, _repo);
          _se->_tse->_map_populate = _index_preload;
#endif
#ifdef USE_FAISS
          _se = new SearchEngine<FaissSE>(dim, _repo);
          if (ad.has("index_type"))
            _se->_tse->_index_key = ad.get("index_type").get<std::string>();
          if (ad.has("train_samples"))
            _se->_tse->_train_samples_size
                = ad.get("train_samples").get<int>();
          if (ad.has("ondisk"))
            _se->_tse->_ondisk = ad.get("ondisk").get<bool>();
          if (ad.has("nprobe"))
            _se->_tse->_nprobe = ad.get("nprobe").get<int>();
#ifdef USE_GPU_FAISS
          if (ad.has("index_gpu"))
            _se->_tse->_gpu = ad.get("index_gpu").get<bool>();
          if (ad.has("index_gpuid"))
            {
              _se->_tse->_gpu = true;
              try
                {
                  int gpuid = ad.get("index_gpuid").get<int>();
                  _se->_tse->_gpuids.push_back(gpuid);
                }
              catch (std::exception &e)
                {
                  try
                    {
                      _se->_tse->_gpuids
                          = ad.get("index_gpuid").get<std::vector<int>>();
                    }
                  catch (std::exception &e)
                    {
                      throw MLLibBadParamException(
                          "wrong type for index_gpuid : id or [id0, id1, "
                          "...]");
                    }
                }
            }
#endif
#endif
          _se->create_index();
        }
    }

    /**
     * \brief create similarity search index
     */
    void create_index()
    {
      if (_se)
        _se->create_index();
    }

    /**
     * \brief build similarity search index
     */
    void build_index()
    {
      if (_se)
        _se->update_index();
    }

    /**
     * \brief remove similarity search index
     */
    void remove_index()
    {
      if (_se)
        _se->remove_index();
    }
#endif

    std::string _repo; /**< model repository. */
    std::string _mlmodel_template_repo = "templates/";
    std::unordered_map<int, std::string>
        _hcorresp;        /**< table of class correspondences. */
    std::string _corresp; /**< file name of the class correspondences (e.g.
                             house / 23) */
    std::string _best_model_filename = "/best_model.txt";

#ifdef USE_SIMSEARCH
#ifdef USE_ANNOY
    SearchEngine<AnnoySE> *_se = nullptr;
#elif USE_FAISS
    SearchEngine<FaissSE> *_se = nullptr;
#endif
    bool _index_preload = false;
#endif

  private:
    void init_repo_dir(const oatpp::Object<DTO::Model> &model_dto, spdlog::logger *logger)
    {
      // auto-creation of model directory
      _repo = model_dto->repository->std_str();

      bool isDir;
      bool exists = fileops::file_exists(_repo, isDir);
      if (exists && !isDir)
        {
          std::string errmsg
              = "file exists with same name as repository " + _repo;
          logger->error(errmsg);
          throw MLLibBadParamException(errmsg);
        }
      if (!exists && model_dto->create_repository)
        fileops::create_dir(_repo, 0775);

      if (!fileops::is_directory_writable(_repo))
        {
          std::string errmsg
              = "destination model directory " + _repo + " is not writable";
          logger->error(errmsg);
          throw MLLibBadParamException(errmsg);
        }

#ifdef USE_SIMSEARCH
      if (model_dto->index_preload)
        _index_preload = true;
#endif
      // auto-install from model archive
      if (model_dto->init)
        {
          std::string compressedf = model_dto->init->std_str();

          // check whether already in the directory
          std::string base_model_fname
              = compressedf.substr(compressedf.find_last_of("/") + 1);
          std::string modelf = _repo + "/" + base_model_fname;
          if (fileops::file_exists(modelf))
            {
              if (logger)
                logger->warn(
                    "Init model {} is already in directory, not fetching it",
                    modelf);
              compressedf = modelf;
            }

          if (compressedf.find("https://") != std::string::npos
              || compressedf.find("http://") != std::string::npos
              || compressedf.find("file://") != std::string::npos)
            {
#ifdef WIN32
              throw MLLibBadParamException("Fetching model archive: "
                                           + compressedf
                                           + " not implemented on Windows");
#else
              int outcode = -1;
              std::string content;
              try
                {
                  if (logger)
                    logger->info("Downloading init model {}", compressedf);
                  httpclient::get_call(compressedf, "GET", outcode, content);
                }
              catch (...)
                {
                  std::string errmsg
                      = "failed fetching model archive: " + compressedf
                        + " with code: " + std::to_string(outcode);
                  logger->error(errmsg);
                  throw MLLibBadParamException(errmsg);
                }
              std::ofstream mof(modelf);
              mof << content;
              mof.close();
              compressedf = modelf;
#endif
            }

          if (fileops::uncompress(compressedf, _repo))
            {
              std::string errmsg = "failed installing model from archive, "
                                   "check 'init' argument "
                                   "to model";
              logger->error(errmsg);
              throw MLLibBadParamException(errmsg);
            }
        }
    }

    void read_config_json(const oatpp::Object<DTO::ServiceCreate> service_create_dto)
    {
      const std::string cf = _repo + "/config.json";
      if (!fileops::file_exists(cf))
        return;
      std::ifstream is(cf);
      std::stringstream jbuf;
      jbuf << is.rdbuf();

      // FIXME(sileht): Replacing the user provided data here doesn't look good
      // to me
      std::shared_ptr<oatpp::data::mapping::ObjectMapper> objectMapper
        = oatpp::parser::json::mapping::ObjectMapper::createShared();
      service_create_dto->parameters = objectMapper->readFromString<oatpp::Object<DTO::Parameters>>(
                  jbuf.str().c_str());
    }
  };
}

#endif
