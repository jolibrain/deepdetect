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
#include "utils/fileops.hpp"
#ifndef WIN32
#include "utils/httpclient.hpp"
#endif
#include "mllibstrategy.h"

namespace dd
{
  class MLModel
  {
  public:
    MLModel() {}

    MLModel(const APIData &ad, APIData &adg,
	    const std::shared_ptr<spdlog::logger> &logger)
      {
	init_repo_dir(ad,logger.get());
	if (ad.has("init"))
	  read_config_json(adg,logger);
      }
    
    MLModel(const APIData &ad)
      {
	init_repo_dir(ad,nullptr);
      }
    
    MLModel(const std::string &repo)
      :_repo(repo) {}


  MLModel(const APIData &ad, const std::string &repo)
    :_repo(repo)
    {
      init_repo_dir(ad,nullptr);
    }

    ~MLModel() {
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
	  std::cerr << "cannot open model corresp file=" << _corresp << std::endl;
	else{
	  std::string line;
	  while(!ff.eof())
	    {
	      std::getline(ff,line);
	      std::string key = line.substr(0,line.find(' '));
	      if (!key.empty())
		{
		  std::string value = line.substr(line.find(' ')+1);
		  _hcorresp.insert(std::pair<int,std::string>(std::stoi(key),value));
		}
	    }
	}
      }
    }

    inline std::string get_hcorresp(const int &i)
      {
	if (_hcorresp.empty())
	  return std::to_string(i);
	else return _hcorresp[i];
      }

#ifdef USE_SIMSEARCH
    /**
     * \brief create similarity search engine
     */
    void create_sim_search(const int &dim)
    {
      if (!_se)
	{
	  _se = new SearchEngine<AnnoySE>(dim,_repo);
	  _se->_tse->_map_populate = _index_preload;
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
    std::unordered_map<int,std::string> _hcorresp; /**< table of class correspondences. */
    std::string _corresp; /**< file name of the class correspondences (e.g. house / 23) */
    std::string _best_model_filename = "/best_model.txt";
    
#ifdef USE_SIMSEARCH
    SearchEngine<AnnoySE> *_se = nullptr;
    bool _index_preload = false;
#endif

  private:
    void init_repo_dir(const APIData &ad,
		       spdlog::logger *logger)
    {
      // auto-creation of model directory
      _repo =  ad.get("repository").get<std::string>();
      bool create = ad.has("create_repository") && ad.get("create_repository").get<bool>();
      bool isDir;
      bool exists = fileops::file_exists(_repo, isDir);
      if (exists && !isDir)
        throw MLLibBadParamException("file exists with same name as repository");
      if (!exists && create)
        fileops::create_dir(_repo,0775);
#ifdef USE_SIMSEARCH
      if (ad.has("index_preload") && ad.get("index_preload").get<bool>())
	_index_preload = true;
#endif
      // auto-install from model archive
      if (ad.has("init"))
	{
	  std::string compressedf = ad.get("init").get<std::string>();

	  // check whether already in the directory
	  std::string base_model_fname = compressedf.substr(compressedf.find_last_of("/") + 1);
	  std::string modelf = _repo + "/" + base_model_fname;
	  if (fileops::file_exists(modelf))
	    {
	      if (logger)
		logger->warn("Init model {} is already in directory, not fetching it",modelf);
	      compressedf = modelf;
	    }
	  
	  if (compressedf.find("https://") != std::string::npos
	      || compressedf.find("http://") != std::string::npos
	      || compressedf.find("file://") != std::string::npos)
	    {
#ifdef WIN32
          throw MLLibBadParamException("Fetching model archive: " + compressedf + " not implemented on Windows");
#else
	      int outcode = -1;
	      std::string content;
	      try
		{
		  if (logger)
		    logger->info("Downloading init model {}",compressedf);
		  httpclient::get_call(compressedf,"GET",outcode,content);
		}
	      catch(...)
		{
		  throw MLLibBadParamException("failed fetching model archive: " + compressedf + " with code: " + std::to_string(outcode));
		}
	      std::ofstream mof(modelf);
	      mof << content;
	      mof.close();
	      compressedf = modelf;
#endif
	    }
	  
	  if (fileops::uncompress(compressedf,_repo))
	    throw MLLibBadParamException("failed installing model from archive, check 'init' argument to model");
	}
    }

    void read_config_json(APIData &adg,
			  const std::shared_ptr<spdlog::logger> &logger)
    {
      const std::string cf = _repo + "/config.json";
      if (!fileops::file_exists(cf))
	return;
      std::ifstream is(cf);
      std::stringstream jbuf;
      jbuf << is.rdbuf();
      rapidjson::Document d;
      d.Parse(jbuf.str().c_str());
      if (d.HasParseError())
	{
	  logger->error("config.json parsing error on string: {}",jbuf.str());
	  throw MLLibBadParamException("Failed parsing config file " + cf);
	}
      APIData adcj;
      try
	{
	  adcj = APIData(d);
	}
      catch(RapidjsonException &e)
	{
	  logger->error("JSON error {}",e.what());
	  throw MLLibBadParamException("Failed converting JSON file to internal data format");
	}
      APIData adcj_parameters = adcj.getobj("parameters");
      adg.add("parameters",adcj_parameters);
    }
  };
}

#endif
