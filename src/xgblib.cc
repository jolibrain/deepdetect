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

#include "xgblib.h"
#include "csvinputfileconn.h"
#include "outputconnectorstrategy.h"
#include <iomanip>
#include <iostream>

//TODO: using xgb structures

namespace xgboost
{
  DMLC_REGISTER_PARAMETER(CLIParam);
}

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::XGBLib(const XGBModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,XGBModel>(cmodel)
  {
    this->_libname = "xgboost";
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::XGBLib(XGBLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,XGBModel>(std::move(cl))
  {
    this->_libname = "xgboost";
    //TODO: copy parameters
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~XGBLib()
  {
    //TODO: delete structures
    /*if (_xgblearn)
      delete _xgblearn;*/
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    //TODO: capture parameters from API for this lib
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    //TODO: anything for clearing the repository
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
									       APIData &out)
  {
    //TODO: any check on model here

    //TODO: mutex if train and predict calls need to be isolated

    TInputConnectorStrategy inputc(this->_inputc);

    //TODO: load data through ::transform
    try
      {
	inputc.transform(ad);
      }
    catch (...)
      {
	throw;
      }

    //TODO: parameters
    std::vector<xgboost::DMatrix*> mats = { inputc._m };
    xgboost::DMatrix *dtrain(inputc._m);
    std::vector<xgboost::DMatrix*> deval;
    std::vector<xgboost::DMatrix*> eval_datasets;
    for (size_t i = 0; i < _params.eval_data_names.size(); ++i) {
      deval.emplace_back(inputc._m); //TODO: test/eval matrix instead
      eval_datasets.push_back(deval.back());
      mats.push_back(deval.back());
    }
    std::vector<std::string> eval_data_names = _params.eval_data_names;
    if (_params.eval_train) {
      eval_datasets.push_back(dtrain);
      eval_data_names.push_back(std::string("train"));
    }
    
    //TODO: initialize the learner
    std::unique_ptr<xgboost::Learner> learner(xgboost::Learner::Create(mats));
    learner->Configure(_params.cfg);
    int version = rabit::LoadCheckPoint(learner.get());
    if (version == 0) {
      // initializ the model if needed.
      if (_params.model_in != "NULL") {
	std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(_params.model_in.c_str(), "r"));
	learner->Load(fi.get());
      } else {
	learner->InitModel();
      }
    }    
    
    //TODO: start training
    //const double start = dmlc::GetTime();
    for (int i = version / 2; i < _params.num_round; ++i) {
      //double elapsed = dmlc::GetTime() - start;
      if (version % 2 == 0) {
	if (_params.silent == 0) {
	  //LOG(INFO) << "boosting round " << i << ", " << elapsed << " sec elapsed";
	}
	learner->UpdateOneIter(i, dtrain);
	if (learner->AllowLazyCheckPoint()) {
	  rabit::LazyCheckPoint(learner.get());
	} else {
	  rabit::CheckPoint(learner.get());
	}
	version += 1;
      }
      CHECK_EQ(version, rabit::VersionNumber());
      std::string res = learner->EvalOneIter(i, eval_datasets, eval_data_names);
      if (rabit::IsDistributed()) {
	if (rabit::GetRank() == 0) {
	  //LOG(TRACKER) << res;
	  LOG(INFO) << res;
	}
      } else {
	if (_params.silent < 2) {
	  //LOG(CONSOLE) << res;
	  LOG(INFO) << res;
	}
      }
      if (_params.save_period != 0 && (i + 1) % _params.save_period == 0) {
	std::ostringstream os;
	os << _params.model_dir << '/'
	   << std::setfill('0') << std::setw(4)
	   << i + 1 << ".model";
	std::unique_ptr<dmlc::Stream> fo(
					 dmlc::Stream::Create(os.str().c_str(), "w"));
	learner->Save(fo.get());
      }
      
      if (learner->AllowLazyCheckPoint()) {
	rabit::LazyCheckPoint(learner.get());
      } else {
	rabit::CheckPoint(learner.get());
      }
      version += 1;
      CHECK_EQ(version, rabit::VersionNumber());
    }
    // always save final round
    if ((_params.save_period == 0 || _params.num_round % _params.save_period != 0) &&
	_params.model_out != "NONE") {
      std::ostringstream os;
      if (_params.model_out == "NULL") {
	os << _params.model_dir << '/'
	   << std::setfill('0') << std::setw(4)
	   << _params.num_round << ".model";
      } else {
	os << _params.model_out;
      }
      std::unique_ptr<dmlc::Stream> fo(
				       dmlc::Stream::Create(os.str().c_str(), "w"));
      learner->Save(fo.get());
    }
    
    /*if (_params.silent == 0) {
      double elapsed = dmlc::GetTime() - start;
      LOG(INFO) << "update end, " << elapsed << " sec in all";
      }*/
     
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    //TODO
  }

  
  template class XGBLib<CSVXGBInputFileConn,SupervisedOutput,XGBModel>;
  
}
