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
    if (_regression)
      this->_mltype = "regression";
    else this->_mltype = "classification";
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
#ifndef CPU_ONLY
    if (ad.has("gpu"))
      _gpu = ad.get("gpu").get<bool>();
#endif
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
    this->_mlmodel.read_from_repository(this->_logger);
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
    // any check on model here
    this->_mlmodel.read_from_repository(this->_logger);
    
    // mutex if train and predict calls need to be isolated
    std::lock_guard<std::mutex> lock(_learner_mutex);
    
    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;    
    APIData cad = ad;
    cad.add("model_repo",this->_mlmodel._repo);
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
#ifndef CPU_ONLY
      if (ad_mllib.has("gpu"))
	_gpu = ad_mllib.get("gpu").get<bool>();
#endif
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
      else if (_objective == "reg:linear" || _objective == "reg:logistic")
	eval_metric = "rmse";
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
      std::string tree_method = "auto";
      double sketch_eps = 0.03;
      double scale_pos_weight = 1.0;
      std::string updater = ""; // was grow_colmaker,prune, now automatic;
      bool refresh_leaf = true;
      std::string process_type = "default";
      std::string dart_sample_type = "uniform";
      std::string dart_normalize_type = "tree";
      double dart_rate_drop = 0.0;
      bool dart_one_drop = false;
      double dart_skip_drop = 0.0;
      
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
	  if (ad_booster.has("tree_method"))
	    tree_method = ad_booster.get("tree_method").get<std::string>();
	  if (ad_booster.has("sketch_eps"))
	    sketch_eps = ad_booster.get("sketch_eps").get<double>();
	  if (ad_booster.has("scale_pos_weight"))
	    scale_pos_weight = ad_booster.get("scale_pos_weight").get<double>();
	  if (ad_booster.has("updater"))
	    updater = ad_booster.get("updater").get<std::string>();
	  if (ad_booster.has("refresh_leaf"))
	    refresh_leaf = ad_booster.get("refresh_leaf").get<bool>();
	  if (ad_booster.has("process_type"))
	    process_type = ad_booster.get("process_type").get<std::string>();
	  if (ad_booster.has("sample_type"))
	    dart_sample_type = ad_booster.get("sample_type").get<std::string>();
	  if (ad_booster.has("normalize_type"))
	    dart_normalize_type = ad_booster.get("normalize_type").get<std::string>();
	  if (ad_booster.has("rate_drop"))
	    dart_rate_drop = ad_booster.get("rate_drop").get<double>();
	  if (ad_booster.has("one_drop"))
	    dart_one_drop = ad_booster.get("one_drop").get<bool>();
	  if (ad_booster.has("skip_drop"))
	    dart_skip_drop = ad_booster.get("skip_drop").get<double>();
	}
#ifndef CPU_ONLY
#ifdef USE_XGBOOST_GPU
      if (_gpu)
	updater = "grow_gpu";
#endif
#endif
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
      add_cfg_param("tree_method",tree_method);
      add_cfg_param("sketch_eps",sketch_eps);
      add_cfg_param("scale_pos_weight",scale_pos_weight);
      if (!updater.empty())
	add_cfg_param("updater",updater);
      add_cfg_param("refresh_leaf",refresh_leaf);
      add_cfg_param("process_type",process_type);
      add_cfg_param("sample_type",dart_sample_type);
      add_cfg_param("normalize_type",dart_normalize_type);
      add_cfg_param("rate_drop",dart_rate_drop);
      add_cfg_param("one_drop",dart_one_drop);
      add_cfg_param("skip_drop",dart_skip_drop);
      
      // data setup
      std::vector<std::shared_ptr<xgboost::DMatrix>> mats = { inputc._m };
      std::shared_ptr<xgboost::DMatrix> dtrain(inputc._m);
      std::vector<std::shared_ptr<xgboost::DMatrix>> deval;
      std::vector<std::shared_ptr<xgboost::DMatrix>> eval_datasets;
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
	  this->_logger->info("initializing model");
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
	    this->_logger->info("boosting round {}",i);
	  }
	  learner->UpdateOneIter(i, dtrain.get());
	  if (learner->AllowLazyCheckPoint()) {
	    rabit::LazyCheckPoint(learner.get());
	  } else {
	    rabit::CheckPoint(learner.get());
	  }
	  version += 1;
	}
	
	// measures for dd
	if (i > 0 && i % test_interval == 0 && !eval_datasets.empty())
	  {
	    APIData meas_out;
	    test(ad,learner,eval_datasets.at(0).get(),meas_out);
	    APIData meas_obj = meas_out.getobj("measure");
	    std::vector<std::string> meas_str = meas_obj.list_keys();
	    for (auto m: meas_str)
	      {
		if (m != "cmdiag" && m != "cmfull" && m != "labels") // do not report confusion matrix in server logs
		  {
		    double mval = meas_obj.get(m).get<double>();
		    this->_logger->info("{}={}",m,mval);
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
		    this->_logger->info("{}=[{}]",m,mdiag_str);
		  }
	      }
	  }
	
	if (_params.save_period != 0 && (i + 1) % _params.save_period == 0) {
	  this->_logger->info("model saving / repo={}",this->_mlmodel._repo);
	  std::ostringstream os;
	  os << this->_mlmodel._repo << '/'
	     << std::setfill('0') << std::setw(4)
	     << i + 1 << ".model";
	  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(os.str().c_str(), "w"));
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
      if ((!this->_tjob_running.load() || _params.save_period == 0 || _params.num_round % _params.save_period != 0)
	  && _params.model_out != "NONE") {
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
      test(ad,learner,inputc._mtest.get(),out);
      
      // prepare model
      this->_mlmodel.read_from_repository(this->_logger);
      this->_mlmodel.read_corresp_file();
      
      // add whatever the input connector needs to transmit out
      inputc.response_params(out);
      
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
	this->_logger->info("loading XGBoost model file={}",model_in);
	std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(model_in.c_str(),"r"));
	_learner->Load(fi.get());
	// we can't read the objective function string name from the xgboost in-memory model,
	// so let's read it from file
	_objective = this->_mlmodel.lookup_objective(model_in,this->_logger);
	if (_objective == "")
	  throw MLLibInternalException("failed to read the objective from XGBoost model file " + model_in);
      }

    // test
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("measure"))
      {
	std::vector<std::shared_ptr<xgboost::DMatrix>> eval_datasets = { inputc._m };
	APIData meas_out;
	std::unique_ptr<xgboost::Learner> learner(_learner);
	test(ad,learner,eval_datasets.at(0).get(),meas_out);
	learner.release();
	meas_out.erase("iteration");
	out.add("measure",meas_out.getobj("measure"));
	return 0;
      }
    
    // predict
    xgboost::HostDeviceVector<float> preds;
    _learner->Configure(_params.cfg);
    _learner->Predict(inputc._m.get(),_params.pred_margin,&preds,_params.ntree_limit);

    // results
    //float loss = 0.0; // XXX: how to acquire loss ?
    int batch_size = preds.Size();
    int nclasses = _nclasses;
    if (_objective == "multi:softprob")
      batch_size /= nclasses;
    else if (_objective == "binary:logistic")
      nclasses--;
    TOutputConnectorStrategy tout;
    std::vector<APIData> vrad;
    for (int j=0;j<batch_size;j++)
      {
	APIData rad;
	rad.add("uri",inputc._ids.at(j));
	rad.add("loss",0.0); // XXX: truely, unreported.
	std::vector<double> probs;
	std::vector<std::string> cats;
	for (int i=0;i<nclasses;i++)
	  {
	    probs.push_back(preds.HostVector().at(j*nclasses+i));
	    cats.push_back(this->_mlmodel.get_hcorresp(i));
	  }
	if (_objective == "binary:logistic")
	  {
	    probs.insert(probs.begin(),1.0-probs.back());
	    cats.insert(cats.begin(),this->_mlmodel.get_hcorresp(1));
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
	out.add("nclasses",nclasses);
      }
    out.add("nclasses",nclasses);
    tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));
    out.add("status",0);
    return 0;
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void XGBLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::test(const APIData &ad,
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
	xgboost::HostDeviceVector<float> out_preds;
	learner->Configure(_params.cfg);
	learner->Predict(dtest,output_margin,&out_preds);

	int nclasses = _nclasses;
	int batch_size = out_preds.Size();
	if (_objective == "multi:softprob")
	  batch_size /= _nclasses;
	else if (_objective == "binary:logistic")
	  nclasses--;
	for (int k=0;k<batch_size;k++)
	  {
	    APIData bad;
	    std::vector<double> predictions;
	    for (int c=0;c<nclasses;c++)
	      {
		predictions.push_back(out_preds.HostVector().at(k*nclasses+c));
	      }
	    if (_objective == "binary:logistic")
	      predictions.insert(predictions.begin(),1.0-predictions.back());
	    bad.add("target",dtest->Info().labels_.HostVector().at(k));
	    bad.add("pred",predictions);
	    ad_res.add(std::to_string(k),bad);
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
