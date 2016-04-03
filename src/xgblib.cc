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
    _nclasses = cl._nclasses;
    _regression = cl._regression;
    _ntargets = cl._ntargets;
    _booster = cl._booster;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~XGBLib()
  {
    if (_learner)
      delete _learner;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
	_regression = true;
	_nclasses = 1;
      }
    if (ad.has("ntargets"))
      _ntargets = ad.get("ntargets").get<int>();
    if (_nclasses == 0)
      throw MLLibBadParamException("number of classes is unknown (nclasses == 0)");
    if (_regression && _ntargets == 0)
      throw MLLibBadParamException("number of regression targets is unknown (ntargets == 0)");
    this->_mlmodel.read_from_repository();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    std::vector<std::string> extensions = {".model"};
    fileops::remove_directory_files(this->_mlmodel._repo,extensions);
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
									       APIData &out)
  {
    //TODO: any check on model here
    this->_mlmodel.read_from_repository();
    
    // mutex if train and predict calls need to be isolated
    std::lock_guard<std::mutex> lock(_learner_mutex);
    
    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;    
    APIData cad = ad;
    try
      {
	inputc.transform(cad);
      }
    catch (...)
      {
	throw;
      }
    
      // parameters
      int num_feature = 0; /**< feature dimension used in boosting, optional. */
      _objective = "multi:softprob";
      double base_score = 0.5;
      std::string eval_metric = "merror";
      int test_interval = 1;
      int seed = 0;
      
      APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
      if (ad_mllib.has("booster"))
	_booster = ad_mllib.get("booster").get<std::string>();
      if (ad_mllib.has("objective"))
	_objective = ad_mllib.get("objective").get<std::string>();
      if (ad_mllib.has("num_feature"))
	num_feature = ad_mllib.get("num_feature").get<int>();
      if (ad_mllib.has("base_score"))
	base_score = ad_mllib.get("base_score").get<double>();
      if (ad_mllib.has("eval_metric"))
	eval_metric = ad_mllib.get("eval_metric").get<std::string>();
      if (ad_mllib.has("seed"))
	seed = ad_mllib.get("seed").get<int>();
      if (ad_mllib.has("iterations"))
	_params.num_round = ad_mllib.get("iterations").get<int>();
      if (ad_mllib.has("test_interval"))
	test_interval = ad_mllib.get("test_interval").get<int>();
      if (ad_mllib.has("save_period"))
	_params.save_period = ad_mllib.get("save_period").get<int>();
      
      _params.eval_train = false;
      _params.eval_data_names.clear();
      _params.eval_data_names.push_back("test");      
      
      add_cfg_param("booster",_booster);
      add_cfg_param("objective",_objective);
      add_cfg_param("num_feature",num_feature);
      add_cfg_param("base_score",base_score);
      add_cfg_param("eval_metric",eval_metric);
      add_cfg_param("seed",seed);
      if (_objective == "multi:softmax")
	throw MLLibBadParamException("use multi:softprob objective instead of multi:softmax");
      if (!_regression && _objective == "multi:softprob")
	add_cfg_param("num_class",_nclasses);
      
      // booster parameters
      double eta = 0.3;
      double gamma = 0.0;
      int max_depth = 6;
      int min_child_weight = 1;
      int max_delta_step = 0;
      double subsample = 1.0;
      double colsample = 1.0;
      double lambda = 1.0;
      double alpha = 0.0;
      double lambda_bias = 0.0; // linear booster parameters

      APIData ad_booster = ad_mllib.getobj("booster_params");
      if (ad_booster.size())
	{
	  if (ad_booster.has("eta"))
	    eta = ad_booster.get("eta").get<double>();
	  if (ad_booster.has("gamma"))
	    gamma = ad_booster.get("gamma").get<double>();
	  if (ad_booster.has("max_depth"))
	    max_depth = ad_booster.get("max_depth").get<int>();
	  if (ad_booster.has("min_child_weight"))
	    min_child_weight = ad_booster.get("min_child_weight").get<int>();
	  if (ad_booster.has("max_delta_step"))
	    max_delta_step = ad_booster.get("max_delta_step").get<int>();
	  if (ad_booster.has("subsample"))
	    subsample = ad_booster.get("subsample").get<double>();
	  if (ad_booster.has("colsample"))
	    colsample = ad_booster.get("colsample").get<double>();
	  if (ad_booster.has("lambda"))
	    lambda = ad_booster.get("lambda").get<double>();
	  if (ad_booster.has("alpha"))
	    alpha = ad_booster.get("alpha").get<double>();
	  if (ad_booster.has("lambda_bias"))
	    lambda_bias = ad_booster.get("lambda_bias").get<double>();
	}
      add_cfg_param("eta",eta);
      add_cfg_param("gamma",gamma);
      add_cfg_param("max_depth",max_depth);
      add_cfg_param("min_child_weight",min_child_weight);
      add_cfg_param("max_delta_step",max_delta_step);
      add_cfg_param("subsample",subsample);
      add_cfg_param("colsample",colsample);
      add_cfg_param("lambda",lambda);
      add_cfg_param("alpha",alpha);
      add_cfg_param("lambda_bias",lambda_bias);
      
      // data setup
      std::vector<xgboost::DMatrix*> mats = { inputc._m };
      xgboost::DMatrix *dtrain(inputc._m);
      std::vector<xgboost::DMatrix*> deval;
      std::vector<xgboost::DMatrix*> eval_datasets;
      for (size_t i = 0; i < _params.eval_data_names.size(); ++i) {
	if (inputc._mtest)
	  {
	    deval.emplace_back(inputc._mtest);
	    eval_datasets.push_back(deval.back());
	    mats.push_back(deval.back());
	  }
      }
      std::vector<std::string> eval_data_names = _params.eval_data_names;
      //TODO: as needed, whether to report accuracy on training set
      if (_params.eval_train) {
	eval_datasets.push_back(dtrain);
	eval_data_names.push_back(std::string("train"));
      }

      //initialize the learner
      this->_tjob_running = true;
      std::unique_ptr<xgboost::Learner> learner(xgboost::Learner::Create(mats));
      learner->Configure(_params.cfg);
      int version = rabit::LoadCheckPoint(learner.get());
      if (version == 0) {
	// initialize the model if needed.
	if (_params.model_in != "NULL") { //TODO: if a model already exists
	  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(_params.model_in.c_str(), "r")); //TODO: update the model path (repo)
	  learner->Load(fi.get());
	} else {
	  LOG(INFO) << "initializing model";
	  learner->InitModel();
	}
      }    
      
      //start training
      for (int i = version / 2; i < _params.num_round; ++i) {
	if (!this->_tjob_running.load())
	  break;
	this->add_meas("iteration",i);
	if (version % 2 == 0) {
	  if (_params.silent == 0) {
	    LOG(INFO) << "boosting round " << i;
	  }
	  learner->UpdateOneIter(i, dtrain);
	  if (learner->AllowLazyCheckPoint()) {
	    rabit::LazyCheckPoint(learner.get());
	  } else {
	    rabit::CheckPoint(learner.get());
	  }
	  version += 1;
	}
	std::cerr << "version=" << version << " / rabit version=" << rabit::VersionNumber() << std::endl;
	//CHECK_EQ(version, rabit::VersionNumber());
	/*std::string res = learner->EvalOneIter(i, eval_datasets, eval_data_names);
	  if (rabit::IsDistributed()) {
	  if (rabit::GetRank() == 0) {
	  //LOG(TRACKER) << res;
	  LOG(INFO) << "res=" << res << std::endl;
	  }
	  } else {
	  std::cerr << "param silent=" << _params.silent << std::endl;
	if (_params.silent < 2) {
	//LOG(CONSOLE) << res;
	LOG(INFO) << "res=" << res << std::endl;
	}
	}*/

	// measures for dd
	if (i > 0 && i % test_interval == 0 && !eval_datasets.empty())
	  {
	    APIData meas_out;
	    test(ad,_objective,learner,eval_datasets.at(0),meas_out);
	    APIData meas_obj = meas_out.getobj("measure");
	    std::vector<std::string> meas_str = meas_obj.list_keys();
	    for (auto m: meas_str)
	      {
		if (m != "cmdiag" && m != "cmfull") // do not report confusion matrix in server logs
		  {
		    double mval = meas_obj.get(m).get<double>();
		    LOG(INFO) << m << "=" << mval;
		    this->add_meas(m,mval);
		    if (!std::isnan(mval)) // if testing occurs once before training even starts, loss is unknown and we don't add it to history.
		      this->add_meas_per_iter(m,mval);
		  }
		else if (m == "cmdiag")
		  {
		    std::vector<double> mdiag = meas_obj.get(m).get<std::vector<double>>();
		    std::string mdiag_str;
		    for (size_t i=0;i<mdiag.size();i++)
		      mdiag_str += std::to_string(i) + ":" + std::to_string(mdiag.at(i)) + " ";
		    LOG(INFO) << m << "=[" << mdiag_str << "]";
		  }
	      }
	  }
	
	LOG(INFO) << "model saving / repo=" << this->_mlmodel._repo << std::endl;;
	if (_params.save_period != 0 && (i + 1) % _params.save_period == 0) {
	  std::ostringstream os;
	  os << this->_mlmodel._repo << '/'
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
	//CHECK_EQ(version, rabit::VersionNumber());
      }
      
      // always save final round
      if (!this->_tjob_running.load() || (_params.save_period == 0 || _params.num_round % _params.save_period != 0) &&
	  _params.model_out != "NONE") {
	std::ostringstream os;
	if (_params.model_out == "NULL") {
	  os << this->_mlmodel._repo << '/'
	     << std::setfill('0') << std::setw(4)
	     << _params.num_round << ".model";
	} else {
	  os << _params.model_out;
	}
	std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(os.str().c_str(), "w"));
	learner->Save(fo.get());
      }

      // bail on forced stop, i.e. not testing the model further.
      if (!this->_tjob_running.load())
	{
	  return 0;
	}
      
      // test
      test(ad,_objective,learner,inputc._mtest,out);
      
      // prepare model
      this->_mlmodel.read_from_repository();
      
      // add whatever the input connector needs to transmit out
      //inputc.response_params(out);

      return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    // mutex since using in-memory learner
    std::lock_guard<std::mutex> lock(_learner_mutex);
    
    // data
    TInputConnectorStrategy inputc(this->_inputc);
    APIData cad = ad;
    try
      {
	inputc.transform(cad);
      }
    catch (...)
      {
	throw;
      }
    
    // load existing model as needed
    if (!_learner)
      {
	_learner = xgboost::Learner::Create({});
	std::string model_in = this->_mlmodel._weights;
	std::cerr << "model file=" << model_in << std::endl;
	std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(model_in.c_str(),"r"));
	_learner->Load(fi.get());
      }

    // test
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("measure"))
      {
	std::vector<xgboost::DMatrix*> eval_datasets = { inputc._m };
	APIData meas_out;
	std::unique_ptr<xgboost::Learner> learner(_learner);
	test(ad,_objective,learner,eval_datasets.at(0),meas_out);
	learner.release();
	meas_out.erase("iteration");
	std::vector<APIData> vad = {meas_out.getobj("measure")};
	out.add("measure",vad);
	return 0;
      }
    
    // predict
    std::vector<float> preds;
    _learner->Predict(inputc._m,_params.pred_margin,&preds,_params.ntree_limit);

    // results
    float loss = 0.0; //TODO: acquire loss ?
    int batch_size = preds.size() / _nclasses;
    TOutputConnectorStrategy tout;
    std::vector<APIData> vrad;
    for (int j=0;j<batch_size;j++)
      {
	APIData rad;
	rad.add("uri",inputc._ids.at(j));
	rad.add("loss",0.0); // XXX: truely, unreported.
	std::vector<double> probs;
	std::vector<std::string> cats;
	for (int i=0;i<_nclasses;i++)
	  {
	    probs.push_back(preds.at(j*_nclasses+i));
	    cats.push_back(this->_mlmodel.get_hcorresp(i));
	  }
	rad.add("probs",probs);
	rad.add("cats",cats);
	vrad.push_back(rad);
      }
    tout.add_results(vrad);
    TOutputConnectorStrategy btout(this->_outputc);
    if (_regression)
      {
	out.add("regression",true);
	out.add("nclasses",_nclasses);
      }
    tout.finalize(ad.getobj("parameters").getobj("output"),out);
    out.add("status",0);
    return 0;
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::test(const APIData &ad,
									       const std::string &objective,
									       std::unique_ptr<xgboost::Learner> &learner,
									       xgboost::DMatrix *dtest,
									       APIData &out)
  {
    if (!dtest)
      return;
    APIData ad_res;
    ad_res.add("iteration",this->get_meas("iteration"));
    //ad_res.add("train_loss",this->get_meas("train_loss")); //TODO: can't acquire the loss yet.
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("measure"))
      {
	int nout = _nclasses;
	if (_regression && _ntargets > 1)
	  nout = _ntargets;
	ad_res.add("nclasses",_nclasses);
	bool output_margin = false;
	std::vector<float> out_preds;
	learner->Predict(dtest,output_margin,&out_preds);

	int nclasses = _nclasses;
	int batch_size = out_preds.size();
	if (objective == "multi:softprob")
	  batch_size /= _nclasses;
	else if (objective == "binary:logistic")
	  nclasses--;
	for (int k=0;k<batch_size;k++)
	  {
	    APIData bad;
	    std::vector<double> predictions;
	    for (int c=0;c<nclasses;c++)
	      {
		predictions.push_back(out_preds.at(k*nclasses+c));
	      }
	    if (objective == "binary:logistic")
	      predictions.insert(predictions.begin(),1.0-predictions.back());
	    bad.add("target",dtest->info().labels.at(k));
	    bad.add("pred",predictions);
	    std::vector<APIData> vad = { bad };
	    ad_res.add(std::to_string(k),vad);
	  }
	std::vector<std::string> clnames;
	for (int i=0;i<nout;i++)
	  clnames.push_back(std::to_string(i));
	ad_res.add("clnames",clnames);
	ad_res.add("batch_size",batch_size);
	if (_regression)
	  ad_res.add("regression",_regression);
      }
    SupervisedOutput::measure(ad_res,ad_out,out);
  }
  
  template class XGBLib<CSVXGBInputFileConn,SupervisedOutput,XGBModel>;
  template class XGBLib<SVMXGBInputFileConn,SupervisedOutput,XGBModel>;
  template class XGBLib<TxtXGBInputFileConn,SupervisedOutput,XGBModel>;
}
