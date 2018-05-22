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

//TODO Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/utils/proto_utils.h>
#pragma GCC diagnostic pop

#include "imginputfileconn.h"
#include "backends/caffe2/caffe2lib.h"
#include "outputconnectorstrategy.h"

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  Caffe2Lib(const Caffe2Model &c2model)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,Caffe2Model>(c2model)
  {
    this->_libname = "caffe2";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  Caffe2Lib(Caffe2Lib &&c2l) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,Caffe2Model>(std::move(c2l))
  {
    this->_libname = "caffe2";

    //TODO: Find a clean way to copy a workspace
    for (auto blobname : c2l._workspace.Blobs()) {
      _workspace.CreateBlob(blobname)->swap(*c2l._workspace.GetBlob(blobname));
    }

    _init_net = c2l._init_net;
    _predict_net = c2l._predict_net;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~Caffe2Lib()
  {
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  instantiate_template(const APIData &ad)
  {
    // - check whether there's a risk of erasing model files
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo, this->_logger))
      throw MLLibBadParamException("error reading or listing Caffe2 models in repository " +
				   this->_mlmodel._repo);
    // - locate template repository
    std::string model_tmpl = ad.get("template").get<std::string>();
    this->_mlmodel._model_template = model_tmpl;
    this->_logger->info("instantiating model template {}",model_tmpl);

    // - copy files to model repository
    std::string source = this->_mlmodel._mlmodel_template_repo + '/' + model_tmpl;
    this->_logger->info("source={}", source);
    this->_logger->info("dest={}", this->_mlmodel._repo);
    auto copy = [&](const std::string &name, std::string &dst) {
      auto src = source + "/" + name;
      dst = this->_mlmodel._repo + "/" + name;
      switch(fileops::copy_file(src, dst)) {
      case 1: throw MLLibBadParamException("failed to locate model template " + src);
      case 2: throw MLLibBadParamException("failed to create model template destination " + dst);
      }
    };
    std::string predict_net_path, init_net_path;
    copy("predict_net.pb", predict_net_path);
    copy("init_net.pb", init_net_path);

    //
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_net_path, &_init_net));
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(predict_net_path, &_predict_net));
    CAFFE_ENFORCE(_workspace.RunNetOnce(_init_net));
    //TODO Create all inexistents external inputs
    _workspace.CreateBlob("gpu_0/data");
    CAFFE_ENFORCE(_workspace.CreateNet(_predict_net));

  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  init_mllib(const APIData &ad)
  {
    if (ad.has("template"))
      instantiate_template(ad);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  clear_mllib(const APIData &)
  {
    std::vector<std::string> extensions({".json"});
    fileops::remove_directory_files(this->_mlmodel._repo,extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  train(const APIData &, APIData &)
  {
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  predict(const APIData &ad, APIData &out)
  {
    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData ad_output = ad.getobj("parameters").getobj("output");
    float confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold")) {
      try {
	confidence_threshold = ad_output.get("confidence_threshold").get<float>();
      } catch(std::exception &e) {
	// try from int
	confidence_threshold = static_cast<float>(ad_output.get("confidence_threshold").get<int>());
      }
    }

    try {
      inputc.transform(ad);
    } catch (std::exception &e) {
      throw;
    }

    // int batch_size = inputc.test_batch_size();
    //TODO Get the batch size and the input tensor from value_info.json
    int batch_size = 1;
    std::vector<APIData> vrad;
    auto &tensor_input = *_workspace.GetBlob("gpu_0/data")->GetMutable<caffe2::TensorCPU>();

    while (true) {
      auto image_count(0);
      try {
	image_count = inputc.get_tensor_test(tensor_input, batch_size);
	if (!image_count)
	  break;
      } catch(std::exception &e) {
	this->_logger->error("exception while filling up network for prediction");
	throw;
      }

      float loss(0);
      std::vector<std::vector<float> > results(image_count);
      auto result_size(0);

      //Running the net
      try {
	CAFFE_ENFORCE(_workspace.RunNetOnce(_predict_net));
	//TODO Get the external output from the predict_net
	auto &tensor = _workspace.GetBlob("gpu_0/softmax")->Get<caffe2::TensorCPU>();
	result_size = tensor.size() / batch_size;
	auto data = tensor.data<float>();
	for (auto &result : results) {
	  result.assign(data, data + result_size);
	  data += result_size;
	}
      } catch(std::exception &e) {
	this->_logger->error("Error while proceeding with supervised prediction forward pass, not enough memory? {}",e.what());
	throw;
      }

      //Adding the results
      for (auto &result : results) {
	APIData rad;
	if (!inputc._ids.empty()) {
	  rad.add("uri", inputc._ids.at(vrad.size()));
	} else {
	  rad.add("uri", std::to_string(vrad.size()));
	}
	rad.add("loss", loss);
	std::vector<double> probs;
	std::vector<std::string> cats;
	for (size_t i = 0; i < result.size(); ++i) {
	  float prob = result[i];
	  if (prob < confidence_threshold)
	    continue;
	  probs.push_back(prob);
	  cats.push_back(this->_mlmodel.get_hcorresp(i));
	}
	rad.add("probs", probs);
	rad.add("cats", cats);
	vrad.push_back(rad);
      }
    } // end prediction loop over batches

    tout.add_results(vrad);
    tout.finalize(ad.getobj("parameters").getobj("output"), out,
		  static_cast<MLModel*>(&this->_mlmodel));
    out.add("status", 0);

    return 0;
  }

  template class Caffe2Lib<ImgCaffe2InputFileConn,SupervisedOutput,Caffe2Model>;
}
