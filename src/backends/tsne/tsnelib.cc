/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#include "tsnelib.h"
#include "csvinputfileconn.h"
#include "outputconnectorstrategy.h"
#include <thread>
#include "utils/utils.hpp"

namespace dd
{
 
  unsigned int hardware_concurrency()
  {
    unsigned int cores = std::thread::hardware_concurrency();
    if (!cores)
      cores = dd_utils::my_hardware_concurrency();
    return cores;
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TSNELib(const TSNEModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TSNEModel>(cmodel)
  {
    this->_libname = "tsne";
  }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TSNELib(TSNELib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TSNEModel>(std::move(cl))
  {
    this->_libname = "tsne";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~TSNELib()
  {
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    (void)ad;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TSNELib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										APIData &out)
  {
    std::lock_guard<std::mutex> lock(_tsne_mutex); // locking per service training call
    
    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;
    APIData cad = ad;
    try
      {
	inputc.transform(cad);
      }
    catch(...)
      {
	throw;
      }
    
    // parameters
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    if (ad_mllib.has("iterations"))
      _iterations = ad_mllib.get("iterations").get<int>();
    if (ad_mllib.has("perplexity"))
      _perplexity = ad_mllib.get("perplexity").get<int>();
    
    // t-sne
    int N = -1;
    int D = -1;
    double *Y = nullptr;
    try
      {
	N = inputc._N;
	D = inputc._D;
	std::cerr << "N=" << N << " / D=" << D << std::endl;
	int num_threads = hardware_concurrency();
	this->_logger->info("Detected {} cores", num_threads);
	Y = new double[N*_no_dims]; // results
	for (int i=0;i<N*_no_dims;i++)
	  Y[i] = 0.0;

	TSNE tsne = TSNE(N,D,_perplexity,_theta);
	tsne.step1(inputc._X.data(),Y,num_threads);
	int test_iter = 50;
	//time_t start = time(0);
	double loss = 0.0;
	for (int iter = 0; iter < _iterations; iter++) {
	  tsne.step2_one_iter(Y,iter,loss,test_iter);
	  this->add_meas("train_loss",loss);
	  this->add_meas_per_iter("train_loss",loss);
	  this->add_meas("iteration", iter);
	}
	
      }
    catch(std::exception &e)
      {
	this->_logger->error(e.what());
	throw; //TODO: MLLib exception
      }

    // capture of results
    TOutputConnectorStrategy tout(this->_outputc);
    std::vector<APIData> vrad;
    for (int i=0;i<N;i++)
      {
	APIData rad;
	rad.add("uri",std::to_string(i)); //TODO: ids
	rad.add("loss",0.0); //TODO: useless ?
	std::vector<double> vals;
	for (int j=0;j<_no_dims;j++)
	  {
	    vals.push_back(Y[i*_no_dims+j]);
	  }
	rad.add("vals",vals);
	vrad.push_back(rad);
      }
    delete[] Y;
    tout.add_results(vrad);
    tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));
    out.add("status",0);
    return 0;
  }

  template class TSNELib<CSVTSNEInputFileConn,UnsupervisedOutput,TSNEModel>;
  template class TSNELib<TxtTSNEInputFileConn,UnsupervisedOutput,TSNEModel>;
}
