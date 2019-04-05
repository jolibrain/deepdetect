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

#include "caffemodel.h"
#include "mllibstrategy.h"
#include "utils/fileops.hpp"
#include "utils/utils.hpp"
#include <exception>
#include <fstream>
#include <iostream>

namespace dd
{
  CaffeModel::CaffeModel(const APIData &ad, APIData &adg,
			 const std::shared_ptr<spdlog::logger> &logger)
    :MLModel(ad,adg,logger)
  {
    if (ad.has("templates"))
      this->_mlmodel_template_repo = ad.get("templates").get<std::string>();
    else this->_mlmodel_template_repo += "caffe/"; // default

    if (ad.has("def"))
      _def = ad.get("def").get<std::string>();
    if (ad.has("trainf"))
      _trainf = ad.get("trainf").get<std::string>();
    if (ad.has("weights"))
      _weights = ad.get("weights").get<std::string>();
    if (ad.has("corresp"))
      _corresp = ad.get("corresp").get<std::string>();
    if (ad.has("solver"))
      _solver = ad.get("solver").get<std::string>();
    if (ad.has("repository"))
      {
       	if (read_from_repository(ad.get("repository").get<std::string>(),spdlog::get("api")))
	  throw MLLibBadParamException("error reading or listing Caffe models in repository " + _repo);
      }
    read_corresp_file();
  }

  CaffeModel::CaffeModel(const APIData &ad)
      :MLModel(ad)
  {
    if (ad.has("templates"))
      this->_mlmodel_template_repo = ad.get("templates").get<std::string>();
    else this->_mlmodel_template_repo += "caffe/"; // default

    if (ad.has("def"))
      _def = ad.get("def").get<std::string>();
    if (ad.has("trainf"))
      _trainf = ad.get("trainf").get<std::string>();
    if (ad.has("weights"))
      _weights = ad.get("weights").get<std::string>();
    if (ad.has("corresp"))
      _corresp = ad.get("corresp").get<std::string>();
    if (ad.has("solver"))
      _solver = ad.get("solver").get<std::string>();
    if (ad.has("repository"))
      {
       	if (read_from_repository(ad.get("repository").get<std::string>(),spdlog::get("api")))
	  throw MLLibBadParamException("error reading or listing Caffe models in repository " + _repo);
      }
    read_corresp_file();
  }
  
  int CaffeModel::read_from_repository(const std::string &repo,
				       const std::shared_ptr<spdlog::logger> &logger,
				       const bool &new_first)
  {
    static std::string deploy = "deploy.prototxt";
    static std::string train = ".prototxt";
    static std::string weights = ".caffemodel";
    static std::string sstate = ".solverstate";
    static std::string corresp = "corresp";
    static std::string solver = "_solver.prototxt";
    static std::string meanf = "mean.binaryproto";
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(repo,true,false,false,lfiles);
    if (e != 0)
      {
	logger->error("error reading or listing caffe models in repository {}",repo);
	return 1;
      }
    std::string deployf,trainf,weightsf,correspf,solverf,sstatef;
    long int state_t=-1, weight_t=-1;
    auto hit = lfiles.begin();
    while(hit!=lfiles.end())
      {
	if ((*hit).find(meanf)!=std::string::npos)
	  {
	    _has_mean_file = true;
	  }
	else if ((*hit).find(sstate)!=std::string::npos)
	  {
	    // stat file to pick the latest one
	    long int st = fileops::file_last_modif((*hit));
	    if (st > state_t)
	      {
		sstatef = (*hit);
		state_t = st;
	      }
	  }
	else if ((*hit).find(weights)!=std::string::npos)
	  {
	    // stat file to pick the latest one
	    long int wt = fileops::file_last_modif((*hit));
	    if (wt > weight_t)
	      {
		weightsf = (*hit);
		weight_t = wt;
	      }
	  }
	else if ((*hit).find(corresp)!=std::string::npos)
	  correspf = (*hit);
	else if ((*hit).find("~")!=std::string::npos 
	    || (*hit).find(".prototxt")==std::string::npos)
	  {
	    ++hit;
	    continue;
	  }
	else if ((*hit).find(deploy)!=std::string::npos)
	  deployf = (*hit);
	else if ((*hit).find(solver)!=std::string::npos)
	  solverf = (*hit);
	else if ((*hit).find(train)!=std::string::npos)
	  trainf = (*hit);
	++hit;
      }
    if (_def.empty())
      _def = deployf;
    if (_trainf.empty())
      _trainf = trainf;
    if (new_first || _weights.empty())
      _weights = weightsf;
    if (_corresp.empty())
      _corresp = correspf;
    if (_solver.empty())
      _solver = solverf;
    if (new_first || _sstate.empty())
      _sstate = sstatef;    
    return 0;
  }

  int CaffeModel::copy_to_target(const std::string &source_repo,
				 const std::string &target_repo,
				 const std::shared_ptr<spdlog::logger> &logger)
  {
    if (target_repo.empty())
      {
	logger->warn("empty target repository, bypassing");
	return 0;
      }
    if (!fileops::create_dir(target_repo,0755)) // create target repo as needed
      logger->info("created target repository {}",target_repo);
    std::string bfile = source_repo + this->_best_model_filename;
    if (fileops::file_exists(bfile))
      {
	std::ifstream inp(bfile);
	if (!inp.is_open())
	  return 1;
	std::string line;
	std::getline(inp,line);
	std::vector<std::string> elts = dd_utils::split(line,':');
	std::string best_caffemodel = "/model_iter_" + elts.at(1) + ".caffemodel";
	if (fileops::copy_file(source_repo + best_caffemodel,target_repo + best_caffemodel))
	  {
	    logger->error("failed copying best model {} to {}",source_repo + best_caffemodel,
			  target_repo + best_caffemodel);
	    return 1;
	  }
	else logger->info("sucessfully copied best model file {}",best_caffemodel);
	std::unordered_set<std::string> lfiles;
	fileops::list_directory(source_repo,true,false,false,lfiles);
	auto hit = lfiles.begin();
	while(hit!=lfiles.end())
	  {
	    if ((*hit).find("prototxt")!=std::string::npos
		|| (*hit).find(".json")!=std::string::npos
              || (*hit).find(".txt")!=std::string::npos
              || (*hit).find("bounds.dat")!=std::string::npos
		|| (*hit).find("vocab.dat")!=std::string::npos)
	      {
		std::vector<std::string> selts = dd_utils::split((*hit),'/');
		fileops::copy_file((*hit),target_repo + '/' + selts.back());
	      }
	    ++hit;
	  }
	logger->info("successfully copied best model files from {} to {}",
		     source_repo,target_repo);
	return 0;
      }
    logger->error("failed finding best model to copy from {} to target repository {}",
		  source_repo,target_repo);
    return 1;
  }

}
