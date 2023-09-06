
// tensorrtlib.cc ---

// Copyright (C) 2019 Jolibrain http://www.jolibrain.com

// Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
#include "outputconnectorstrategy.h"
#include "tensorrtlib.h"
#include "utils/apitools.h"
#include "tensorrtinputconns.h"
#include "tensorrtcalibrator.hpp"
#include "NvInferPlugin.h"
#include "../parsers/onnx/NvOnnxParser.h"
#include "protoUtils.h"
#include <cuda_runtime_api.h>
#include <string>
#include "dto/service_predict.hpp"
#ifdef USE_CUDA_CV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#include "utils/bbox.hpp"
#include "models/yolo.hpp"

namespace dd
{

  static TRTLogger trtLogger;

  static std::string dtype_to_str(nvinfer1::DataType dtype)
  {
    switch (dtype)
      {
      case nvinfer1::DataType::kFLOAT:
        return "fp32";
      case nvinfer1::DataType::kHALF:
        return "fp16";
      case nvinfer1::DataType::kINT32:
        return "int32";
      case nvinfer1::DataType::kINT8:
        return "int8";
      default:
        throw MLLibInternalException("Unsupported datatype: "
                                     + std::to_string(int(dtype)));
      }
  }

  static int findEngineBS(std::string repo, std::string engineFileName,
                          std::string arch, nvinfer1::DataType dtype)
  {
    std::unordered_set<std::string> lfiles;
    fileops::list_directory(repo, true, false, false, lfiles);
    for (std::string s : lfiles)
      {
        // Ommiting directory name
        auto fstart = s.find_last_of("/");
        if (fstart == std::string::npos)
          fstart = 0;

        if (s.find(engineFileName + "_arch" + arch + "_" + dtype_to_str(dtype),
                   fstart)
            != std::string::npos)
          {
            std::string bs_str;
            for (auto it = s.crbegin(); it != s.crend(); ++it)
              {
                if (isdigit(*it))
                  bs_str += (*it);
                else
                  break;
              }
            std::reverse(bs_str.begin(), bs_str.end());
            if (bs_str.length() == 0)
              return -1;
            return std::stoi(bs_str);
          }
      }
    return -1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::TensorRTLib(const TensorRTModel &cmodel)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TensorRTModel>(cmodel)
  {
    this->_libname = "tensorrt";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::TensorRTLib(TensorRTLib &&tl) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TensorRTModel>(std::move(tl))
  {
    this->_libname = "tensorrt";
    _nclasses = tl._nclasses;
    _width = tl._width;
    _height = tl._height;
    _dla = tl._dla;
    _datatype = tl._datatype;
    _max_batch_size = tl._max_batch_size;
    _max_workspace_size = tl._max_workspace_size;
    _results_height = tl._results_height;
    _builder = tl._builder;
    _builderc = tl._builderc;
    _engineFileName = tl._engineFileName;
    _readEngine = tl._readEngine;
    _writeEngine = tl._writeEngine;
    _arch = tl._arch;
    _gpuid = tl._gpuid;
    _TRTContextReady = tl._TRTContextReady;
    _buffers = tl._buffers;
    _bbox = tl._bbox;
    _ctc = tl._ctc;
    _timeserie = tl._timeserie;
    _regression = tl._regression;
    _need_nms = tl._need_nms;
    _template = tl._template;
    _inputIndex = tl._inputIndex;
    _outputIndex0 = tl._outputIndex0;
    _outputIndex1 = tl._outputIndex1;
    _explicit_batch = tl._explicit_batch;
    _floatOut = tl._floatOut;
    _keepCount = tl._keepCount;
    _dims = tl._dims;
    _error_recorder = tl._error_recorder;
    _calibrator = tl._calibrator;
    _engine = tl._engine;
    _builder = tl._builder;
    _context = tl._context;
    _builderc = tl._builderc;
    _runtime = tl._runtime;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::~TensorRTLib()
  {
    // Delete objects in the correct order
    _calibrator = nullptr;
    _context = nullptr;
    _engine = nullptr;
    _builderc = nullptr;
    _builder = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::init_mllib(const APIData &ad)
  {
    trtLogger.setLogger(this->_logger);
    initLibNvInferPlugins(&trtLogger, "");
    _runtime = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(trtLogger));
    _error_recorder.reset(new TRTErrorRecorder(this->_logger));
    _runtime->setErrorRecorder(_error_recorder.get());

    if (ad.has("tensorRTEngineFile"))
      _engineFileName = ad.get("tensorRTEngineFile").get<std::string>();
    if (ad.has("readEngine"))
      _readEngine = ad.get("readEngine").get<bool>();
    if (ad.has("writeEngine"))
      _writeEngine = ad.get("writeEngine").get<bool>();

    if (ad.has("maxWorkspaceSize"))
      {
        _max_workspace_size = ad.get("maxWorkspaceSize").get<int>();
        size_t meg = 1 << 20;
        _max_workspace_size *= meg;
        this->_logger->info("setting max workspace size to {}",
                            _max_workspace_size);
      }
    if (ad.has("maxBatchSize"))
      {
        int nmbs = ad.get("maxBatchSize").get<int>();
        _max_batch_size = nmbs;
        this->_logger->info("setting max batch size to {}", _max_batch_size);
      }
    if (ad.has("nclasses"))
      {
        _nclasses = ad.get("nclasses").get<int>();
      }

    if (ad.has("dla"))
      _dla = ad.get("dla").get<int>();

    if (ad.has("datatype"))
      {
        std::string datatype = ad.get("datatype").get<std::string>();
        if (datatype == "fp32")
          _datatype = nvinfer1::DataType::kFLOAT;
        else if (datatype == "fp16")
          _datatype = nvinfer1::DataType::kHALF;
        else if (datatype == "int32")
          _datatype = nvinfer1::DataType::kINT32;
        else if (datatype == "int8")
          _datatype = nvinfer1::DataType::kINT8;
      }

    if (this->_mlmodel.is_caffe_source()
        && caffe_proto::isRefinedet(this->_mlmodel._def))
      {
        if (_datatype != nvinfer1::DataType::kFLOAT)
          {
            this->_logger->warn("refinedet detected : forcing fp32");
            _datatype = nvinfer1::DataType::kFLOAT;
          }
      }

    if (ad.has("gpuid"))
      _gpuid = ad.get("gpuid").get<int>();
    cudaSetDevice(_gpuid);

    model_type(this->_mlmodel._def, this->_mltype);
    if (this->_mlmodel._source_type.empty())
      {
        throw MLLibBadParamException(
            "cannot find caffe or onnx model in repository, make sure there's "
            "a net_tensorRT.proto or net_tensorRT.onnx file in repository"
            + this->_mlmodel._repo);
      }

    if (ad.has("template"))
      {
        _template = ad.get("template").get<std::string>();
        this->_logger->info("Model template is {}", _template);

        if (_template == "yolox")
          {
            this->_mltype = "detection";
            _need_nms = true;
          }
        else
          throw MLLibBadParamException("Unknown template " + _template);
      }

    _builder = std::shared_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(trtLogger));
    _builderc = std::shared_ptr<nvinfer1::IBuilderConfig>(
        _builder->createBuilderConfig());

    if (_dla != -1)
      {
        if (_builder->getNbDLACores() == 0)
          this->_logger->info("Trying to use DLA core on a platform  that "
                              "doesn't have any DLA cores");
        else
          {
            if (_datatype == nvinfer1::DataType::kINT32)
              {
                this->_logger->info("asked for int32 on dla : forcing int8");
                _datatype = nvinfer1::DataType::kINT8;
              }
            else if (_datatype == nvinfer1::DataType::kFLOAT)
              {
                this->_logger->info(
                    "asked for float32 on dla : forcing float16");
                _datatype = nvinfer1::DataType::kHALF;
              }
            _builderc->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            if (_datatype == nvinfer1::DataType::kINT8)
              {
                _builderc->setFlag(nvinfer1::BuilderFlag::kINT8);
              }
            else
              {
                _builderc->setFlag(nvinfer1::BuilderFlag::kFP16);
              }
            _builderc->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            _builderc->setDLACore(_dla);
            _builderc->setFlag(
                nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            _builderc->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
            _builderc->setFlag(
                nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
          }
      }
    else
      {
        if (_datatype == nvinfer1::DataType::kHALF)
          {
            if (!_builder->platformHasFastFp16())
              {
                _builderc->setFlag(nvinfer1::BuilderFlag::kFP16);
                this->_logger->info("Setting FP16 mode");
              }
            else
              this->_logger->info("Platform does not have fast FP16 mode");
          }
        else if (_datatype == nvinfer1::DataType::kINT8)
          {
            if (_builder->platformHasFastInt8())
              {
                _builderc->setFlag(nvinfer1::BuilderFlag::kINT8);
                this->_logger->info("Setting INT8 mode");
              }
            else
              this->_logger->info("Platform does not have fast INT8 mode");
          }
        else
          {
            // default is TF32 (aka 10-bit mantissas round-up)
          }
      }

    // check on the input size
    if (!_timeserie && this->_mlmodel.is_caffe_source())
      {
        if (_width == 0 || _height == 0)
          {
            this->_logger->info("trying to determine the input size...");
            if (caffe_proto::findInputDimensions(this->_mlmodel._def, _width,
                                                 _height))
              {
                this->_logger->info("found {}x{} as input size", _width,
                                    _height);
              }
            else
              throw MLLibBadParamException("Could not detect the input size, "
                                           "specify it from API instead");
          }
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    if (!_buffers.empty())
      {
        cudaFree(_buffers.at(_inputIndex));
        cudaFree(_buffers.at(_outputIndex0));
        if (_bbox)
          cudaFree(_buffers.at(_outputIndex1));
      }

    // remove compiled model files.
    std::vector<std::string> extensions
        = { "TRTengine", "net_tensorRT.proto", "calibration_table" };
    fileops::remove_directory_files(this->_mlmodel._repo, extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                  TMLModel>::train(const APIData &ad, APIData &out)
  {
    this->_logger->warn("Training not supported on tensorRT backend");
    (void)ad;
    (void)out;
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  nvinfer1::ICudaEngine *
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::read_engine_from_caffe(const std::string &out_blob)
  {
    int fixcode = caffe_proto::fixProto(this->_mlmodel._repo + "/"
                                            + "net_tensorRT.proto",
                                        this->_mlmodel._def);
    switch (fixcode)
      {
      case 1:
        this->_logger->error("TRT backend could not open model prototxt");
        break;
      case 2:
        this->_logger->error("TRT backend  could not write "
                             "transformed model prototxt");
        break;
      default:
        break;
      }

    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        _builder->createNetworkV2(0U));
    nvcaffeparser1::ICaffeParser *caffeParser
        = nvcaffeparser1::createCaffeParser();

    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor
        = caffeParser->parse(
            std::string(this->_mlmodel._repo + "/" + "net_tensorRT.proto")
                .c_str(),
            this->_mlmodel._weights.c_str(), *network, _datatype);
    if (!blobNameToTensor)
      throw MLLibInternalException("Error while parsing caffe model "
                                   "for conversion to TensorRT");

    auto bloboutput = blobNameToTensor->find(out_blob.c_str());
    if (!bloboutput)
      throw MLLibBadParamException("Cannot find output layer " + out_blob);
    network->markOutput(*bloboutput);

    if (out_blob == "detection_out")
      network->markOutput(*blobNameToTensor->find("keep_count"));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    _builder->setMaxBatchSize(_max_batch_size);
#pragma GCC diagnostic pop

    _builderc->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                  _max_workspace_size);
    if (_calibrator)
      _builderc->setInt8Calibrator(_calibrator.get());

    network->getLayer(0)->setPrecision(nvinfer1::DataType::kFLOAT);

    nvinfer1::ILayer *outl = NULL;
    int idx = network->getNbLayers() - 1;
    while (outl == NULL)
      {
        nvinfer1::ILayer *l = network->getLayer(idx);
        if (strcmp(l->getName(), out_blob.c_str()) == 0)
          {
            outl = l;
            break;
          }
        idx--;
      }
    // force output to be float32
    outl->setPrecision(nvinfer1::DataType::kFLOAT);
    nvinfer1::IHostMemory *n
        = _builder->buildSerializedNetwork(*network, *_builderc);
    return _runtime->deserializeCudaEngine(n->data(), n->size());
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  nvinfer1::ICudaEngine *
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::read_engine_from_onnx()
  {
    const auto explicitBatch
        = 1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        _builder->createNetworkV2(explicitBatch));
    _explicit_batch = true;

    nvonnxparser::IParser *onnxParser
        = nvonnxparser::createParser(*network, trtLogger);
    onnxParser->parseFromFile(this->_mlmodel._model.c_str(),
                              int(nvinfer1::ILogger::Severity::kWARNING));

    if (onnxParser->getNbErrors() != 0)
      {
        for (int i = 0; i < onnxParser->getNbErrors(); ++i)
          {
            this->_logger->error(onnxParser->getError(i)->desc());
          }
        throw MLLibInternalException(
            "Error while parsing onnx model for conversion to "
            "TensorRT");
      }

    if (_calibrator)
      _builderc->setInt8Calibrator(_calibrator.get());

    // TODO check with onnx models dynamic shape
    this->_logger->warn("Onnx model: max batch size not used");
    _builderc->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                  _max_workspace_size);

    nvinfer1::IHostMemory *n
        = _builder->buildSerializedNetwork(*network, *_builderc);

    if (n == nullptr)
      throw MLLibInternalException("Could not build model: "
                                   + this->_mlmodel._model);
    return _runtime->deserializeCudaEngine(n->data(), n->size());
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  oatpp::Object<DTO::PredictBody>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::predict(const APIData &ad)
  {
    std::lock_guard<std::mutex> lock(
        _net_mutex); // no concurrent calls since the net is not
                     // re-instantiated

    oatpp::Object<DTO::ServicePredict> predict_dto;
    // XXX: until everything is DTO, we consider the two cases:
    // - either ad embeds a DTO, so we just have to retrieve it
    // - or it's an APIData that must be converted to DTO
    if (ad.has("dto"))
      {
        // cast to ServicePredict...
        auto any = ad.get("dto").get<oatpp::Any>();
        predict_dto = oatpp::Object<DTO::ServicePredict>(
            std::static_pointer_cast<typename DTO::ServicePredict>(any->ptr));
      }
    else
      {
        predict_dto = ad.createSharedDTO<DTO::ServicePredict>();

        if (ad.has("chain") && ad.get("chain").get<bool>())
          predict_dto->_chain = true;
        if (ad.has("data_raw_img"))
          predict_dto->_data_raw_img
              = ad.get("data_raw_img").get<std::vector<cv::Mat>>();
#ifdef USE_CUDA_CV
        if (ad.has("data_raw_img_cuda"))
          predict_dto->_data_raw_img_cuda
              = ad.get("data_raw_img_cuda")
                    .get<std::vector<cv::cuda::GpuMat>>();
#endif
        if (ad.has("ids"))
          predict_dto->_ids = ad.get("ids").get<std::vector<std::string>>();
        if (ad.has("meta_uris"))
          predict_dto->_meta_uris
              = ad.get("meta_uris").get<std::vector<std::string>>();
        if (ad.has("index_uris"))
          predict_dto->_index_uris
              = ad.get("index_uris").get<std::vector<std::string>>();
      }

    if (predict_dto->parameters->mllib->gpuid->_ids.size() == 0)
      throw MLLibBadParamException("empty gpuid vector");
    if (predict_dto->parameters->mllib->gpuid->_ids.size() > 1)
      throw MLLibBadParamException(
          "TensorRT: Multi-GPU inference is not applicable");

    _gpuid = predict_dto->parameters->mllib->gpuid->_ids[0];
    cudaSetDevice(_gpuid);

    auto output_params = predict_dto->parameters->output;

    std::string out_blob = "prob";
    std::string extract_layer = predict_dto->parameters->mllib->extract_layer;
    bool calibration = predict_dto->parameters->mllib->calibration;

    TInputConnectorStrategy inputc(this->_inputc);

    if (!_TRTContextReady)
      {
        // detect architecture
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, _gpuid);
        _arch = std::to_string(prop.major) + std::to_string(prop.minor);
        this->_logger->info("GPU {} architecture = compute_{}", _gpuid, _arch);

        _bbox = output_params->bbox;
        _regression = output_params->regression;
        _ctc = output_params->ctc;

        if (_ctc)
          {
            if (output_params->blank_label != -1)
              throw MLLibBadParamException(
                  "blank_label not yet implemented over tensorRT "
                  "backend");
          }

        if (_bbox == true)
          out_blob = "detection_out";
        else if (_ctc == true)
          {
            out_blob = "probs";
            throw MLLibBadParamException(
                "ocr not yet implemented over tensorRT backend");
          }
        else if (_timeserie)
          {
            out_blob = "rnn_pred";
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else if (_regression)
          out_blob = "pred";
        else if (!extract_layer.empty())
          out_blob = extract_layer;

        if (_nclasses == 0 && this->_mlmodel.is_caffe_source())
          {
            this->_logger->info("trying to determine number of classes...");
            _nclasses = caffe_proto::findNClasses(this->_mlmodel._def, _bbox);
            if (_nclasses < 0)
              throw MLLibBadParamException("failed detecting the number of "
                                           "classes, specify it through "
                                           "API with nclasses");
            this->_logger->info("found {} classes", _nclasses);
          }

        if (_bbox)
          {
            if (this->_mlmodel.is_onnx_source())
              _results_height
                  = onnx_proto::findBBoxCount(this->_mlmodel._model, out_blob);
            else if (!this->_mlmodel._def.empty())
              _results_height
                  = caffe_proto::findBBoxCount(this->_mlmodel._def);
          }

        if (_nclasses <= 0)
          this->_logger->error("could not determine number of classes");

        bool engineRead = false;
        std::string engine_path = this->_mlmodel._repo + "/" + _engineFileName
                                  + "_arch" + _arch + "_"
                                  + dtype_to_str(_datatype) + "_bs"
                                  + std::to_string(_max_batch_size);

        if (_readEngine)
          {
            int bs = findEngineBS(this->_mlmodel._repo, _engineFileName, _arch,
                                  _datatype);
            if (bs != _max_batch_size && bs != -1)
              {
                throw MLLibBadParamException(
                    "found existing engine with max_batch_size "
                    + std::to_string(bs) + " instead of "
                    + std::to_string(_max_batch_size)
                    + " / either delete it or set your maxBatchSize to "
                    + std::to_string(bs));
              }
            std::ifstream file(engine_path, std::ios::binary);
            if (file.good())
              {
                std::vector<char> trtModelStream;
                size_t size{ 0 };
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream.resize(size);
                file.read(trtModelStream.data(), size);
                file.close();

                auto *errors = _runtime->getErrorRecorder();
                errors->clear();
                _engine = std::shared_ptr<nvinfer1::ICudaEngine>(
                    _runtime->deserializeCudaEngine(trtModelStream.data(),
                                                    trtModelStream.size()));

                bool shouldRecompile = false;
                for (int i = 0; i < errors->getNbErrors(); ++i)
                  {
                    std::string desc = errors->getErrorDesc(i);
                    if (desc.find("Version tag does not match")
                        != std::string::npos)
                      {
                        this->_logger->warn(
                            "Engine is outdated and will be recompiled");
                        shouldRecompile = true;
                      }
                  }

                if (!shouldRecompile)
                  {
                    if (_engine == nullptr)
                      throw MLLibInternalException(
                          "Engine could not be deserialized");

                    engineRead = true;
                  }
              }
            else
              {
                this->_logger->warn(
                    "Could not read engine at {}, will recompile.",
                    engine_path);
              }
          }

        if (!engineRead)
          {
            nvinfer1::ICudaEngine *le = nullptr;
            if (_datatype == nvinfer1::DataType::kINT8)
              {
                if (calibration)
                  {
                    try
                      {
                        inputc.transform(predict_dto);
                      }
                    catch (...)
                      {
                        throw;
                      }
                  }

                bool calibrate_from_cache = !calibration;
                if (calibrate_from_cache)
                  this->_logger->info(
                      "Setting up the int8 calibrator using cache");
                else
                  this->_logger->info(
                      "Setting up the int8 calibrator using test data");
                _calibrator.reset(new TRTCalibrator<TInputConnectorStrategy>(
                    &inputc, this->_mlmodel._repo, _max_batch_size,
                    calibrate_from_cache, this->_logger));
              }

            if (this->_mlmodel._model.find("net_tensorRT.proto")
                    != std::string::npos
                || !this->_mlmodel._def.empty())
              {
                le = read_engine_from_caffe(out_blob);
              }
            else if (this->_mlmodel._source_type == "onnx")
              {
                le = read_engine_from_onnx();
              }
            else
              {
                throw MLLibInternalException(
                    "No model to parse for conversion to TensorRT");
              }

            _engine = std::shared_ptr<nvinfer1::ICudaEngine>(le);

            if (_writeEngine)
              {
                std::ofstream p(engine_path, std::ios::binary);
                nvinfer1::IHostMemory *trtModelStream = _engine->serialize();
                p.write(reinterpret_cast<const char *>(trtModelStream->data()),
                        trtModelStream->size());
              }

            // once the engine is built, calibrator is not needed anymore
            _calibrator = nullptr;
          }
        else
          {
            if (this->_mlmodel.is_onnx_source())
              _explicit_batch = true;
          }

        _context = std::shared_ptr<nvinfer1::IExecutionContext>(
            _engine->createExecutionContext());
        _TRTContextReady = true;

        try
          {
            _inputName = _engine->getIOTensorName(0);
            if (out_blob == "last")
              _outputName0
                  = _engine->getIOTensorName(_engine->getNbIOTensors() - 1);
            else
              _outputName0 = out_blob;

            // Get dimensions
            _dims = _engine->getTensorShape(_outputName0.c_str());
            if (_dims.nbDims >= 2)
              {
                this->_logger->info(
                    "detected output dimensions: [{}, {} {} {}]", _dims.d[0],
                    _dims.d[1], _dims.nbDims > 2 ? _dims.d[2] : 0,
                    _dims.nbDims > 3 ? _dims.d[3] : 0);

                if (_explicit_batch)
                  {
                    this->_logger->warn(
                        "Explicit batch: set max batch size to "
                        "model batch size {}",
                        _dims.d[0]);
                    _max_batch_size = _dims.d[0];
                  }
              }
          }
        catch (...)
          {
            throw MLLibInternalException("Cannot find or bind output layer "
                                         + out_blob);
          }

        if (_first_predict)
          {
            _first_predict = false;

            if (_bbox)
              {
                if (_dims.nbDims < 3)
                  throw MLLibBadParamException(
                      "Bbox model requires 3 output dimensions, found "
                      + std::to_string(_dims.nbDims));

                _outputName1 = "keep_count";
                _buffers.resize(3);
                int det_out_size
                    = _max_batch_size * _results_height * _dims.d[2];
                _floatOut.resize(det_out_size);
                _keepCount.resize(_max_batch_size);
                if (inputc._bw)
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * inputc._height * inputc._width
                                 * sizeof(float));
                else
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * 3 * inputc._height
                                 * inputc._width * sizeof(float));
                cudaMalloc(&_buffers.data()[_outputIndex0],
                           det_out_size * sizeof(float));
                cudaMalloc(&_buffers.data()[_outputIndex1],
                           _max_batch_size * sizeof(int));
              }
            else if (_ctc)
              {
                throw MLLibBadParamException(
                    "ocr not yet implemented over tensorRT backend");
              }
            else if (_timeserie)
              {
                throw MLLibBadParamException(
                    "timeseries not yet implemented over tensorRT backend");
              }
            // GAN / raw output
            else if (!extract_layer.empty())
              {
                _buffers.resize(2);
                if (_dims.nbDims == 4)
                  _floatOut.resize(_max_batch_size * _dims.d[1] * _dims.d[2]
                                   * _dims.d[3]);
                else
                  throw MLLibBadParamException(
                      "raw/image output model requires 4 output dimensions");
                if (inputc._bw)
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * inputc._height * inputc._width
                                 * sizeof(float));
                else
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * 3 * inputc._height
                                 * inputc._width * sizeof(float));
                cudaMalloc(&_buffers.data()[_outputIndex0],
                           _max_batch_size * _dims.d[1] * _dims.d[2]
                               * _dims.d[3] * sizeof(float));
              }
            else // classification / regression
              {
                _buffers.resize(2);
                _floatOut.resize(_max_batch_size * this->_nclasses);
                if (inputc._bw)
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * inputc._height * inputc._width
                                 * sizeof(float));
                else
                  cudaMalloc(&_buffers.data()[_inputIndex],
                             _max_batch_size * 3 * inputc._height
                                 * inputc._width * sizeof(float));
                cudaMalloc(&_buffers.data()[_outputIndex0],
                           _max_batch_size * _nclasses * sizeof(float));
              }
          }
      }

    cudaSetDevice(_gpuid);
    cudaStream_t cstream;
    cudaStreamCreate(&cstream);

    TOutputConnectorStrategy tout(this->_outputc);
    this->_stats.transform_start();
#ifdef USE_CUDA_CV
    inputc._cuda_buf = static_cast<float *>(_buffers.at(_inputIndex));
    auto cv_stream = cv::cuda::StreamAccessor::wrapStream(cstream);
    inputc._cuda_stream = &cv_stream;
#endif

    try
      {
        inputc.transform(predict_dto);
      }
    catch (...)
      {
        throw;
      }
    this->_stats.transform_end();

    this->_stats.inc_inference_count(inputc._batch_size);

    if (!_timeserie && this->_mlmodel.is_caffe_source())
      {
        if (inputc.width() != _width || inputc.height() != _height)
          {
            throw MLLibBadParamException(
                "Model height and/or width differ from the input data "
                "size. Input data size should be "
                + std::to_string(_width) + "x" + std::to_string(_height));
          }
      }

    int idoffset = 0;
    std::vector<APIData> vrad;
    std::vector<UnsupervisedResult> unsup_results;

    bool enqueue_success = false;
    while (true)
      {

        int num_processed = inputc.process_batch(_max_batch_size);
        if (num_processed == 0)
          break;

        // some models don't support dynamic batch size (ex: onnx)
        // this can lead to undetected bad predictions
        if (num_processed > _dims.d[0])
          throw MLLibBadParamException(
              "Trying to process " + std::to_string(num_processed)
              + " element, but the model has a maximum batch size of "
              + std::to_string(_dims.d[0]));

        try
          {
#ifdef USE_CUDA_CV
            if (!inputc._cuda)
#endif
              {
                if (inputc._bw)
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
                else
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * 3 * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
              }

            if (!_explicit_batch)
              {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                enqueue_success = _context->enqueue(
                    num_processed, _buffers.data(), cstream, nullptr);
#pragma GCC diagnostic pop
              }
            else
              {
                _context->setTensorAddress(_inputName.c_str(),
                                           _buffers.data()[_inputIndex]);
                _context->setTensorAddress(_outputName0.c_str(),
                                           _buffers.data()[_outputIndex0]);
                if (_buffers.size() >= 3)
                  {
                    _context->setTensorAddress(_outputName1.c_str(),
                                               _buffers.data()[_outputIndex1]);
                  }

                enqueue_success = _context->enqueueV3(cstream);
              }
            if (!enqueue_success)
              throw MLLibInternalException("Failed TRT enqueue call");

            if (_bbox)
              {
                cudaMemcpyAsync(_floatOut.data(),
                                _buffers.data()[_outputIndex0],
                                _floatOut.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, cstream);
                cudaMemcpyAsync(_keepCount.data(),
                                _buffers.data()[_outputIndex1],
                                num_processed * sizeof(int),
                                cudaMemcpyDeviceToHost, cstream);
                cudaStreamSynchronize(cstream);
              }
            else if (_ctc)
              {
                throw MLLibBadParamException(
                    "ocr not yet implemented over tensorRT backend");
              }
            else if (_timeserie)
              {
                throw MLLibBadParamException(
                    "timeseries not yet implemented over tensorRT backend");
              }
            // GAN/raw output
            else if (!extract_layer.empty())
              {
                cudaMemcpyAsync(_floatOut.data(),
                                _buffers.data()[_outputIndex0],
                                _floatOut.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, cstream);
                cudaStreamSynchronize(cstream);
              }
            else // classification / regression
              {
                cudaMemcpyAsync(_floatOut.data(),
                                _buffers.data()[_outputIndex0],
                                num_processed * _nclasses * sizeof(float),
                                cudaMemcpyDeviceToHost, cstream);
                cudaStreamSynchronize(cstream);
              }
          }
        catch (std::exception &e)
          {
            this->_logger->error("Error while processing forward TRT pass, "
                                 "not enough memory ? {}",
                                 e.what());
            throw;
          }

        std::vector<double> probs;
        std::vector<std::string> cats;
        std::vector<APIData> bboxes;
        std::vector<APIData> series;

        if (_bbox)
          {
            int top_k = _results_height;
            if (output_params->top_k > 0)
              top_k = output_params->top_k;
            const float *outr = _floatOut.data();

            // preproc yolox
            std::vector<float> yolo_out;
            if (_template == "yolox")
              {
                yolo_out = yolo_utils::parse_yolo_output(
                    _floatOut, num_processed, _results_height, _dims.d[2],
                    _nclasses, inputc._width, inputc._height);
                outr = yolo_out.data();
              };

            const int det_size = 7;

            for (int j = 0; j < num_processed; j++)
              {
                int k = 0;
                std::vector<double> probs;
                std::vector<std::string> cats;
                std::vector<APIData> bboxes;
                APIData rad;
                std::string uri = inputc._ids.at(idoffset + j);
                auto bit = inputc._imgs_size.find(uri);
                int rows = 1;
                int cols = 1;
                if (bit != inputc._imgs_size.end())
                  {
                    // original image size
                    rows = (*bit).second.first;
                    cols = (*bit).second.second;
                  }
                else
                  {
                    this->_logger->error(
                        "couldn't find original image size for {}", uri);
                  }
                bool leave = false;
                int curi = -1;

                while (true && k < top_k)
                  {
                    if (!_need_nms && output_params->best_bbox > 0
                        && bboxes.size() >= static_cast<size_t>(
                               output_params->best_bbox))
                      break;

                    if (outr[0] == -1)
                      {
                        // skipping invalid detection
                        this->_logger->error("skipping invalid detection");
                        outr += det_size;
                        leave = true;
                        break;
                      }
                    std::vector<float> detection(outr, outr + det_size);
                    if (curi == -1)
                      curi = detection[0]; // first pass
                    else if (curi != detection[0])
                      break; // this belongs to next image
                    ++k;
                    outr += det_size;

                    if (detection[2] < output_params->confidence_threshold)
                      continue;

                    // Fix border of bboxes
                    detection[3]
                        = std::max(((float)detection[3]), 0.0f) * (cols - 1);
                    detection[4]
                        = std::max(((float)detection[4]), 0.0f) * (rows - 1);
                    detection[5]
                        = std::min(((float)detection[5]), 1.0f) * (cols - 1);
                    detection[6]
                        = std::min(((float)detection[6]), 1.0f) * (rows - 1);

                    probs.push_back(detection[2]);
                    cats.push_back(this->_mlmodel.get_hcorresp(detection[1]));
                    APIData ad_bbox;
                    ad_bbox.add("xmin", static_cast<double>(detection[3]));
                    ad_bbox.add("ymin", static_cast<double>(detection[4]));
                    ad_bbox.add("xmax", static_cast<double>(detection[5]));
                    ad_bbox.add("ymax", static_cast<double>(detection[6]));
                    bboxes.push_back(ad_bbox);
                  }

                if (_need_nms)
                  {
                    // We assume that bboxes are already sorted in model output
                    bbox_utils::nms_sorted_bboxes(
                        bboxes, probs, cats,
                        (double)output_params->nms_threshold,
                        (int)output_params->best_bbox);
                  }

                if (leave)
                  continue;
                rad.add("uri", uri);
                rad.add("loss", 0.0); // XXX: unused
                rad.add("probs", probs);
                rad.add("cats", cats);
                rad.add("bboxes", bboxes);
                vrad.push_back(rad);
              }
          }

        else if (_ctc)
          {
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else if (_timeserie)
          {
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else if (!extract_layer.empty())
          {
            for (int j = 0; j < num_processed; j++)
              {
                UnsupervisedResult result;
                if (!inputc._ids.empty())
                  result._uri = inputc._ids.at(idoffset + j);
                else
                  result._uri = std::to_string(idoffset + j);
                result._loss = 0.0;

                if (output_params->image)
                  {
                    size_t img_chan = size_t(_dims.d[1]);
                    size_t img_width = size_t(_dims.d[2]),
                           img_height = size_t(_dims.d[3]);
                    auto cv_type = img_chan == 3 ? CV_8UC3 : CV_8UC1;
                    cv::Mat vals_mat(img_width, img_height, cv_type);

                    size_t chan_offset = img_width * img_height;

                    for (size_t y = 0; y < img_height; ++y)
                      {
                        for (size_t x = 0; x < img_width; ++x)
                          {
                            if (cv_type == CV_8UC3)
                              {
                                vals_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(
                                    static_cast<int8_t>(
                                        (_floatOut[2 * chan_offset
                                                   + y * img_width + x]
                                         + 1)
                                        * 255.0 / 2.0),
                                    static_cast<int8_t>(
                                        (_floatOut[1 * chan_offset
                                                   + y * img_width + x]
                                         + 1)
                                        * 255.0 / 2.0),
                                    static_cast<int8_t>(
                                        (_floatOut[0 * chan_offset
                                                   + y * img_width + x]
                                         + 1)
                                        * 255.0 / 2.0));
                              }
                            else
                              {
                                vals_mat.at<int8_t>(y, x)
                                    = static_cast<int8_t>(
                                        (_floatOut[y * img_width + x] + 1)
                                        * 255.0 / 2.0);
                              }
                          }
                      }

                    result._images.push_back(vals_mat);
                  }
                else
                  {
                    result._vals = std::vector<double>(_floatOut.begin(),
                                                       _floatOut.end());
                  }
                unsup_results.push_back(std::move(result));
              }
          }
        else // classification / regression
          {
            for (int j = 0; j < num_processed; j++)
              {
                APIData rad;
                if (!inputc._ids.empty())
                  rad.add("uri", inputc._ids.at(idoffset + j));
                else
                  rad.add("uri", std::to_string(idoffset + j));
                rad.add("loss", 0.0);
                std::vector<double> probs;
                std::vector<std::string> cats;

                for (int i = 0; i < _nclasses; i++)
                  {
                    double prob = _floatOut.at(j * _nclasses + i);
                    if (prob < output_params->confidence_threshold
                        && !_regression)
                      continue;
                    probs.push_back(prob);
                    cats.push_back(this->_mlmodel.get_hcorresp(i));
                  }

                rad.add("probs", probs);
                rad.add("cats", cats);
                vrad.push_back(rad);
              }
          }
        idoffset += num_processed;
      }

    cudaStreamDestroy(cstream);

    oatpp::Object<DTO::PredictBody> out_dto;
    OutputConnectorConfig conf;
    if (extract_layer.empty())
      {
        tout.add_results(vrad);
        conf._nclasses = this->_nclasses;
        if (_bbox)
          conf._has_bbox = true;
        if (_regression)
          conf._regression = true;
        out_dto = tout.finalize(predict_dto->parameters->output, conf,
                                static_cast<MLModel *>(&this->_mlmodel));
      }
    else
      {
        UnsupervisedOutput unsupo;
        unsupo.set_results(std::move(unsup_results));
        out_dto = unsupo.finalize(predict_dto->parameters->output, conf,
                                  static_cast<MLModel *>(&this->_mlmodel));
      }

    if (predict_dto->_chain)
      {
        if (typeid(inputc) == typeid(ImgTensorRTInputFileConn))
          {
            auto *img_ic
                = reinterpret_cast<ImgTensorRTInputFileConn *>(&inputc);
#ifdef USE_CUDA_CV
            if (!img_ic->_cuda_images.empty())
              {
                if (img_ic->_orig_images.empty())
                  out_dto->_chain_input._cuda_imgs = img_ic->_cuda_orig_images;
                else
                  out_dto->_chain_input._cuda_imgs = img_ic->_cuda_images;
              }
            else
#endif
              {
                if (!img_ic->_orig_images.empty())
                  out_dto->_chain_input._imgs = img_ic->_orig_images;
                else
                  out_dto->_chain_input._imgs = img_ic->_images;
              }
            out_dto->_chain_input._img_sizes = img_ic->_images_size;
          }
      }

    // out_dto->status = 0;
    return out_dto;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::model_type(const std::string &param_file,
                                         std::string &mltype)
  {
    std::ifstream paramf(param_file);
    std::stringstream content;
    content << paramf.rdbuf();

    std::size_t found_detection = content.str().find("DetectionOutput");
    if (found_detection != std::string::npos)
      {
        mltype = "detection";
        return;
      }
    std::size_t found_ocr = content.str().find("ContinuationIndicator");
    if (found_ocr != std::string::npos)
      {
        mltype = "ctc";
        return;
      }
    mltype = "classification";
  }

  template class TensorRTLib<ImgTensorRTInputFileConn, SupervisedOutput,
                             TensorRTModel>;
}
