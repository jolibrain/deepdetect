/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Authors: Louis Jean <ljean@etud.insa-toulouse.fr>
 *           Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "torchmodel.h"
#include "utils/utils.hpp"

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

namespace dd
{
  int TorchModel::read_from_repository(
      const std::shared_ptr<spdlog::logger> &logger)
  {

    const std::string weights = ".ptw";
    const std::string native = ".npt";
    const std::string traced = ".pt";
    const std::string corresp = "corresp";
    // solver. may lead to _solver.prototxt when generated from caffe generator
    // we save solver states as solver-##.pt where ## is iteration number
    const std::string sstate = "solver-";
    const std::string proto = "proto";

    std::unordered_set<std::string> files;
    int err = fileops::list_directory(_repo, true, false, false, files);

    if (err != 0)
      {
        logger->error("Listing pytorch models failed");
        return 1;
      }

    std::string tracedf, weightsf, correspf, sstatef, protof, nativef;
    int traced_t = -1, weights_t = -1, corresp_t = -1, sstate_t = -1,
        proto_t = -1, native_t = -1;

    for (const auto &file : files)
      {
        long int lm = fileops::file_last_modif(file);
        if (file.find(sstate) != std::string::npos)
          {
            if (sstate_t < lm)
              {
                sstatef = file;
                sstate_t = lm;
              }
          }
        else if (file.find(weights) != std::string::npos)
          {
            if (weights_t < lm)
              {
                weightsf = file;
                weights_t = lm;
              }
          }
        else if (file.find(traced) != std::string::npos)
          {
            if (traced_t < lm)
              {
                tracedf = file;
                traced_t = lm;
              }
          }
        else if (file.find(native) != std::string::npos)
          {
            if (native_t < lm)
              {
                nativef = file;
                native_t = lm;
              }
          }
        else if (file.find(corresp) != std::string::npos)
          {
            if (corresp_t < lm)
              {
                correspf = file;
                corresp_t = lm;
              }
          }
        else if (file.find(proto) != std::string::npos)
          {
            if (proto_t < lm)
              {
                protof = file;
                proto_t = lm;
              }
          }
      }

    _traced = tracedf;
    _weights = weightsf;
    _corresp = correspf;
    _sstate = sstatef;
    _proto = protof;
    _native = nativef;

    return 0;
  }

  int TorchModel::copy_to_target(const std::string &source_repo,
                                 const std::string &target_repo,
                                 const std::shared_ptr<spdlog::logger> &logger)
  {
    if (target_repo.empty())
      {
        logger->warn("empty string given as target repository, bypassing");
        return 0;
      }

    if (!fileops::create_dir(target_repo,
                             0755)) // create target repo as needed
      logger->info("created target repository {}", target_repo);

    std::string bfile = source_repo + this->_best_model_filename;
    if (fileops::file_exists(bfile))
      {
        std::ifstream inp(bfile);
        if (!inp.is_open())
          return 1;
        std::string line;
        std::string best_checkpoint;
        while (std::getline(inp, line))
          {
            std::vector<std::string> elts = dd_utils::split(line, ':');
            if (elts.at(0).find("iteration") != std::string::npos)
              {
                best_checkpoint = "/checkpoint-" + elts.at(1);
                break;
              }
          }
        if (best_checkpoint.empty())
          {
            logger->error(
                "best model file does not contains key \"iteration\": {}",
                bfile);
            return 1;
          }

        // Copy checkpoint
        bool checkpoint_copied = false;
        if (!fileops::copy_file(source_repo + best_checkpoint + ".pt",
                                target_repo + best_checkpoint + ".pt"))
          {
            logger->info("sucessfully copied best model file {}.pt",
                         source_repo + best_checkpoint);
            checkpoint_copied = true;
          }
        if (!fileops::copy_file(source_repo + best_checkpoint + ".npt",
                                target_repo + best_checkpoint + ".npt"))
          {
            logger->info("sucessfully copied best model file {}.npt",
                         source_repo + best_checkpoint);
            checkpoint_copied = true;
          }
        if (!fileops::copy_file(source_repo + best_checkpoint + ".ptw",
                                target_repo + best_checkpoint + ".ptw"))
          {
            logger->info("sucessfully copied best model file {}.ptw",
                         source_repo + best_checkpoint);
            checkpoint_copied = true;
          }

        if (!checkpoint_copied)
          {
            logger->error("failed copying best model {} to {} (extensions: "
                          ".pt, .npt, .ptw)",
                          source_repo + best_checkpoint,
                          target_repo + best_checkpoint);
            return 1;
          }

        // copy other files
        std::unordered_set<std::string> lfiles;
        fileops::list_directory(source_repo, true, false, false, lfiles);
        auto hit = lfiles.begin();
        while (hit != lfiles.end())
          {
            if ((*hit).find("prototxt") != std::string::npos
                || (*hit).find(".json") != std::string::npos
                || (*hit).find(".txt") != std::string::npos
                || (*hit).find("bounds.dat") != std::string::npos
                || (*hit).find("vocab.dat") != std::string::npos)
              {
                std::vector<std::string> selts = dd_utils::split((*hit), '/');
                fileops::copy_file((*hit), target_repo + '/' + selts.back());
                logger->info("successfully copied model file {} to {}", (*hit),
                             target_repo + '/' + selts.back());
              }
            ++hit;
          }

        logger->info("successfully copied best model files from {} to {}",
                     source_repo, target_repo);

        update_config_json_parameters(target_repo, logger);

        return 0;
      }
    // else if best model file does not exist
    logger->error(
        "failed finding best model to copy from {} to target repository {}",
        source_repo, target_repo);
    return 1;
  }

  void TorchModel::update_config_json_parameters(
      const std::string &target_repo,
      const std::shared_ptr<spdlog::logger> &logger)
  {
    // parse config.json and model.json
    std::string config_path = target_repo + "/config.json";
    std::string model_path = target_repo + "/model.json";
    std::ifstream ifs_config(config_path.c_str(), std::ios::binary);
    if (!ifs_config.is_open())
      {
        logger->error("could not find config file {} for export update",
                      config_path);
        return;
      }
    std::stringstream config_sstr;
    config_sstr << ifs_config.rdbuf();
    ifs_config.close();
    std::ifstream ifs_model(model_path.c_str(), std::ios::binary);
    if (!ifs_model.is_open())
      {
        logger->error("could not find model file {} for export update",
                      config_path);
        return;
      }
    std::stringstream model_sstr;
    model_sstr << ifs_model.rdbuf();
    ifs_model.close();

    rapidjson::Document d_config;
    d_config.Parse<rapidjson::kParseNanAndInfFlag>(config_sstr.str().c_str());
    rapidjson::Document d_model;
    d_model.Parse<rapidjson::kParseNanAndInfFlag>(model_sstr.str().c_str());

    //- repository
    d_config["model"]["repository"].SetString(target_repo.c_str(),
                                              d_config.GetAllocator());

    //- crop_size
    auto d_config_input = d_config["parameters"]["input"].GetObject();
    auto d_model_mllib = d_model["parameters"]["mllib"].GetObject();
    if (d_model_mllib.HasMember("crop_size"))
      {
        try
          {
            int crop_size = d_model_mllib["crop_size"].GetInt();
            if (crop_size > 0)
              {
                d_config_input["width"].SetInt(crop_size);
                d_config_input["height"].SetInt(crop_size);
              }
          }
        catch (RapidjsonException &e)
          {
          }
      }
    //- db
    try
      {
        if (d_config_input.HasMember("db"))
          d_config_input["db"].SetBool(false);
        if (d_model_mllib.HasMember("db"))
          d_model_mllib["db"].SetBool(false);
      }
    catch (RapidjsonException &e)
      {
      }

    // save updated config.json
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                      rapidjson::UTF8<>, rapidjson::CrtAllocator,
                      rapidjson::kWriteNanAndInfFlag>
        writer(buffer);
    bool done = d_config.Accept(writer);
    if (!done)
      throw DataConversionException("JSON rendering failed");
    std::string config_str = buffer.GetString();
    std::ofstream config_out(config_path.c_str(), std::ios::out
                                                      | std::ios::binary
                                                      | std::ios::trunc);
    config_out << config_str;
    config_out.close();
    logger->info("successfully updated {}", config_path);
  }
}
