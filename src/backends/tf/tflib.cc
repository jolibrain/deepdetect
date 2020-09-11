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

#include <string>
#include "tflib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TFLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TFLib(
      const TFModel &cmodel)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TFModel>(
          cmodel)
  {
    this->_libname = "tensorflow";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TFLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TFLib(
      TFLib &&cl) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TFModel>(
          std::move(cl))
  {
    this->_libname = "tensorflow";
    _nclasses = cl._nclasses;
    _regression = cl._regression;
    _ntargets = cl._ntargets;
    _inputLayer = cl._inputLayer;
    _outputLayer = cl._outputLayer;
    _inputFlag = cl._inputFlag;
    this->_mltype = "classification";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TFLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TFLib()
  {
    std::lock_guard<std::mutex> lock(_net_mutex);
    if (_session)
      {
        _session->Close();
        _session.reset();
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TFLib<TInputConnectorStrategy, TOutputConnectorStrategy,
             TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
        _regression = true; // XXX: unsupported
        _nclasses = 1;
      }
    // setting the value of Input Layer for Tensorflow graph
    if (ad.has("inputlayer"))
      {
        _inputLayer = ad.get("inputlayer").get<std::string>();
      }
    // setting the final Output Layer for Tensorflow graph
    if (ad.has("outputlayer"))
      {
        _outputLayer = ad.get("outputlayer").get<std::string>();
      }
    if (ad.has("input_flag"))
      {
        _inputFlag = ad.getobj("input_flag");
      }
    if (ad.has("ntargets")) // XXX: unsupported
      _ntargets = ad.get("ntargets").get<int>();
    if (_nclasses == 0)
      throw MLLibBadParamException(
          "number of classes is unknown (nclasses == 0)");
    if (_regression && _ntargets == 0)
      throw MLLibBadParamException(
          "number of regression targets is unknown (ntargets == 0)");
    this->_mlmodel.read_from_repository(this->_mlmodel._repo, this->_logger);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TFLib<TInputConnectorStrategy, TOutputConnectorStrategy,
             TMLModel>::clear_mllib(const APIData &ad)
  {
    // NOT IMPLEMENTED
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TFLib<TInputConnectorStrategy, TOutputConnectorStrategy,
             TMLModel>::tf_concat(const std::vector<tensorflow::Tensor> &dv,
                                  std::vector<tensorflow::Tensor> &vtfinputs)
  {
    /*
      as incredible as it seems, code below is the bloated tf way of
      concatenating tensors to produce the intput to the neural net graph.
    */
    int rbatch_size = dv.size();
    auto root = tensorflow::Scope::NewRootScope();
    std::string concat_name = "concatenated";
    std::vector<tensorflow::Input> ops_inputs;
    for (int i = 0; i < rbatch_size; i++)
      ops_inputs.push_back(std::move(tensorflow::Input(dv[i])));
    tensorflow::gtl::ArraySlice<tensorflow::Input> ipl(&ops_inputs[0],
                                                       ops_inputs.size());
    tensorflow::InputList toil(ipl);
    auto concatout
        = tensorflow::ops::Concat(root.WithOpName(concat_name), toil, 0);
    std::unique_ptr<tensorflow::Session> concat_session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::GraphDef graph;
    root.ToGraphDef(&graph);
    concat_session->Create(graph);
    tensorflow::Status concat_run_status
        = concat_session->Run({}, { concat_name }, {}, &vtfinputs);
    if (!concat_run_status.ok())
      {
        std::cout << concat_run_status.ToString() << std::endl;
        throw MLLibInternalException(concat_run_status.ToString());
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TFLib<TInputConnectorStrategy, TOutputConnectorStrategy,
            TMLModel>::train(const APIData &ad, APIData &out)
  {
    // NOT IMPLEMENTED
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void
  TFLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::test(
      const APIData &ad, APIData &out)
  {
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (!ad_out.has("measure"))
      {
        APIData ad_res;
        SupervisedOutput::measure(ad_res, ad_out, out);
        return;
      }

    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout(this->_outputc);
    APIData cad = ad;
    cad.add("model_repo", this->_mlmodel._repo);
    try
      {
        inputc.transform(cad);
      }
    catch (std::exception &e)
      {
        throw;
      }

    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    APIData ad_output = ad.getobj("parameters").getobj("output");
    int batch_size = inputc.batch_size();
    if (ad_mllib.has("test_batch_size"))
      {
        batch_size = ad_mllib.get("test_batch_size").get<int>();
      }

    tensorflow::GraphDef graph_def;
    std::string graphFile = this->_mlmodel._graphName;
    if (graphFile.empty())
      throw MLLibBadParamException(
          "No pre-trained model found in model repository");
    this->_logger->info("test: using graphFile dir={}", graphFile);
    // Loading the graph to the given variable
    tensorflow::Status graphLoadedStatus
        = ReadBinaryProto(tensorflow::Env::Default(), graphFile, &graph_def);

    if (!graphLoadedStatus.ok())
      {
        this->_logger->error("failed loading tensorflow graph with status={}",
                             graphLoadedStatus.ToString());
        throw MLLibBadParamException(
            "failed loading tensorflow graph with status="
            + graphLoadedStatus.ToString());
      }
    std::string inputLayer = _inputLayer;
    std::string outputLayer = _outputLayer;
    if (inputLayer.empty())
      {
        inputLayer = graph_def.node(0).name();
        this->_logger->info("testing: using input layer={}", inputLayer);
      }
    if (outputLayer.empty())
      {
        outputLayer = graph_def.node(graph_def.node_size() - 1).name();
        this->_logger->info("testing: using output layer={}", outputLayer);
      }

    // creating a session with the graph
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto &config = options.config;
    config.mutable_gpu_options()->set_allow_growth(
        true); // default is we prevent tf from holding all memory across all
               // GPUs
    auto session = std::unique_ptr<tensorflow::Session>(
        tensorflow::NewSession(options));
    tensorflow::Status session_create_status = session->Create(graph_def);

    if (!session_create_status.ok())
      {
        std::cout << session_create_status.ToString() << std::endl;
        throw MLLibInternalException(session_create_status.ToString());
      }

    // vector for storing  the outputAPI of the file
    APIData ad_res;
    ad_res.add("nclasses", _nclasses);
    int tresults = 0;

    inputc.reset_dv();
    int idoffset = 0;
    while (true)
      {
        std::vector<int> dv_labels = inputc._test_labels;
        std::vector<tensorflow::Tensor> dv = inputc.get_dv(batch_size);
        if (dv.empty())
          break;
        std::vector<tensorflow::Tensor> vtfinputs;
        if (dv.size() > 1)
          tf_concat(dv, vtfinputs);
        else
          vtfinputs = dv;

        // running the loded graph and saving the generated output
        std::vector<tensorflow::Tensor>
            finalOutput; // To save the final Output generated by the
                         // tensorflow
        tensorflow::Status run_status
            = session->Run({ { inputLayer, *(vtfinputs.begin()) } },
                           { outputLayer }, {}, &finalOutput);
        if (!run_status.ok())
          {
            std::cout << run_status.ToString() << std::endl;
            throw MLLibInternalException(
                run_status.ToString()); // XXX: does not separate bad param
                                        // from internal errors
          }
        tensorflow::Tensor output = std::move(finalOutput.at(0));

        auto scores = output.flat<float>();
        std::vector<double> predictions;
        for (size_t i = 0; i < dv.size(); i++)
          {
            APIData bad;
            for (int c = 0; c < _nclasses; c++)
              predictions.push_back(scores(i * _nclasses + c));
            double target = dv_labels.at(i);
            bad.add("target", target);
            bad.add("pred", predictions);
            ad_res.add(std::to_string(tresults + i), bad);
          }
        tresults += dv.size();
      } // end prediction loop over batches

    std::vector<std::string> clnames;
    for (int i = 0; i < _nclasses; i++)
      clnames.push_back(this->_mlmodel.get_hcorresp(i));
    ad_res.add("clnames", clnames);
    ad_res.add("batch_size", tresults);

    SupervisedOutput::measure(ad_res, ad_out, out);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TFLib<TInputConnectorStrategy, TOutputConnectorStrategy,
            TMLModel>::predict(const APIData &ad, APIData &out)
  {
    // TF sessions support concurrent calls, however server design enforces
    // preference for using batches to max out resources instead of
    // cumulated calls that may overflow the resources.
    // This policy may be subjected to futur changes.
    std::lock_guard<std::mutex> lock(_net_mutex);

    APIData ad_output = ad.getobj("parameters").getobj("output");
    if (ad_output.has("measure"))
      {
        test(ad, out);
        APIData out_meas = out.getobj("measure");
        out.add("measure", out_meas);
        return 0;
      }
    double confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold"))
      confidence_threshold
          = ad_output.get("confidence_threshold").get<double>();
    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData cad = ad;
    cad.add("model_repo", this->_mlmodel._repo);
    try
      {
        inputc.transform(cad);
      }
    catch (std::exception &e)
      {
        throw;
      }

    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    int batch_size = inputc.batch_size();
    if (ad_mllib.has("test_batch_size"))
      batch_size = ad_mllib.get("test_batch_size").get<int>();

    std::string extract_layer;
    if (ad_mllib.has("extract_layer")
        && !ad_mllib.get("extract_layer").get<std::string>().empty())
      {
        _outputLayer = ad_mllib.get("extract_layer").get<std::string>();
        extract_layer = _outputLayer;
      }

    if (!_session)
      {
        tensorflow::GraphDef graph_def;
        std::string graphFile = this->_mlmodel._graphName;
        if (graphFile.empty())
          throw MLLibBadParamException(
              "No pre-trained model found in model repository");
        this->_logger->info("predict: using graphFile dir={}", graphFile);
        // Loading the graph to the given variable
        tensorflow::Status graphLoadedStatus = ReadBinaryProto(
            tensorflow::Env::Default(), graphFile, &graph_def);

        if (!graphLoadedStatus.ok())
          {
            this->_logger->error(
                "failed loading tensorflow graph with status={}",
                graphLoadedStatus.ToString());
            throw MLLibBadParamException(
                "failed loading tensorflow graph with status="
                + graphLoadedStatus.ToString());
          }

        /*for (int l=0;l<graph_def.node_size();l++)
          {
            std::cerr << graph_def.node(l).name() << std::endl;
            }*/

        if (_inputLayer.empty())
          {
            _inputLayer = graph_def.node(0).name();
            this->_logger->info("using input layer={}", _inputLayer);
          }
        if (_outputLayer.empty())
          {
            _outputLayer = graph_def.node(graph_def.node_size() - 1).name();
            this->_logger->info("using output layer={}", _outputLayer);
          }
        // tensorflow::graph::SetDefaultDevice(device, &graph_def);

        // creating a session with the graph
        tensorflow::SessionOptions options;
        tensorflow::ConfigProto &config = options.config;
        config.mutable_gpu_options()->set_allow_growth(
            true); // default is we prevent tf from holding all memory across
                   // all GPUs
        _session = std::unique_ptr<tensorflow::Session>(
            tensorflow::NewSession(options));
        tensorflow::Status session_create_status = _session->Create(graph_def);

        if (!session_create_status.ok())
          {
            std::cout << session_create_status.ToString() << std::endl;
            _session = nullptr;
            throw MLLibInternalException(session_create_status.ToString());
          }
      }

    // vector for storing  the outputAPI of the file
    std::vector<APIData> vrad;
    inputc.reset_dv();
    int idoffset = 0;
    while (true)
      {
        std::vector<tensorflow::Tensor> dv = inputc.get_dv(batch_size);
        if (dv.empty())
          break;
        std::vector<tensorflow::Tensor> vtfinputs;
        if (dv.size() > 1)
          tf_concat(dv, vtfinputs);
        else
          vtfinputs = dv;

        // other input variables
        std::pair<std::string, tensorflow::Tensor> othertfinputs;
        std::vector<std::string> lkeys = _inputFlag.list_keys();
        bool has_input_vars = false;
        for (auto k : lkeys)
          {
            tensorflow::Tensor ivar(tensorflow::DT_BOOL,
                                    tensorflow::TensorShape());
            ivar.scalar<bool>()() = _inputFlag.get(k).get<bool>();
            othertfinputs.first = k;
            othertfinputs.second = ivar;
            has_input_vars = true;
            break; // a single key for now, may have to use ClientSession for
                   // another scheme
          }

        // running the loded graph and saving the generated output
        std::vector<tensorflow::Tensor>
            finalOutput; // To save the final output generated by the
                         // tensorflow
        tensorflow::Status run_status;
        if (has_input_vars)
          run_status = _session->Run(
              { { _inputLayer, *(vtfinputs.begin()) }, othertfinputs },
              { _outputLayer }, {}, &finalOutput);
        else
          run_status = _session->Run({ { _inputLayer, *(vtfinputs.begin()) } },
                                     { _outputLayer }, {}, &finalOutput);
        if (!run_status.ok())
          {
            std::cout << run_status.ToString() << std::endl;
            throw MLLibInternalException(
                run_status.ToString()); // TODO: separate bad param and
                                        // internal errors
          }
        tensorflow::Tensor output = std::move(finalOutput.at(0));

        APIData rad;
        if (extract_layer.empty()) // supervised setting
          {
            auto scores = output.flat<float>();
            for (size_t i = 0; i < dv.size(); i++)
              {
                rad.add("uri", inputc._ids.at(idoffset + i));
                std::vector<double> probs;
                std::vector<std::string> cats;
                for (int c = 0; c < _nclasses; c++)
                  {
                    // std::cerr << "score=" << scores(c) << " / c=" << c <<
                    // std::endl;
                    double prob = scores(i * _nclasses + c);
                    if (prob < confidence_threshold)
                      continue;
                    probs.push_back(prob);
                    cats.push_back(this->_mlmodel.get_hcorresp(c));
                  }
                rad.add("probs", probs);
                rad.add("cats", cats);
                rad.add("loss", 0.0);
                vrad.push_back(rad);
              }
            idoffset += dv.size();
          }
        else // unsupervised
          {
            auto layer_vals = output.flat<float>();
            int embedding_size
                = layer_vals.size() / static_cast<float>(dv.size());
            int offset = 0;
            for (size_t i = 0; i < dv.size(); i++)
              {
                std::vector<double> vals;
                vals.reserve(embedding_size); // layer_vals.size());
                for (int c = 0; c < embedding_size; c++)
                  vals.push_back(layer_vals.data()[offset + c]);
                rad.add("uri", inputc._ids.at(idoffset + i));
                rad.add("vals", vals);
                vrad.push_back(rad);
                offset += embedding_size;
              }
            idoffset += dv.size();
          }
      } // end prediction loop over batches
    tout.add_results(vrad);
    out.add("nclasses", _nclasses);
    tout.finalize(ad.getobj("parameters").getobj("output"), out,
                  static_cast<MLModel *>(&this->_mlmodel));
    out.add("status", 0);
    return 0;
  }

  template class TFLib<ImgTFInputFileConn, SupervisedOutput, TFModel>;
  template class TFLib<ImgTFInputFileConn, UnsupervisedOutput, TFModel>;
}
