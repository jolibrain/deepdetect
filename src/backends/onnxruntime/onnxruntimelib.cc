/**
 * DeepDetect
 * Copyright (c) 2026 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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

#include "outputconnectorstrategy.h"
#include "onnxruntimelib.h"
#include "dto/service_predict.hpp"
#include "utils/bbox.hpp"
#include "../tensorrt/models/yolo.hpp"
#include <algorithm>

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
#include <cuda_runtime_api.h>
#endif

#ifdef USE_CUDA_CV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace dd
{
  // --------------------------------------------------------------------------
  // Constructors & Destructor
  // --------------------------------------------------------------------------

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                 TMLModel>::ONNXRuntimeLib(const ONNXRuntimeModel &model)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              ONNXRuntimeModel>(model)
  {
    this->_libname = "onnxruntime";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                 TMLModel>::ONNXRuntimeLib(ONNXRuntimeLib &&other) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              ONNXRuntimeModel>(std::move(other)),
        _env_ptr(std::move(other._env_ptr)),
        _session_options(std::move(other._session_options)),
        _session(std::move(other._session)),
        _input_node_names(std::move(other._input_node_names)),
        _output_node_names(std::move(other._output_node_names)),
        _input_node_names_ptr(std::move(other._input_node_names_ptr)),
        _output_node_names_ptr(std::move(other._output_node_names_ptr)),
        _input_dims(std::move(other._input_dims)),
        _output_dims(std::move(other._output_dims))
  {
    this->_libname = "onnxruntime";
    this->_nclasses = other._nclasses;
    this->_width = other._width;
    this->_height = other._height;
    this->_max_batch_size = other._max_batch_size;
    this->_gpuid = other._gpuid;
    this->_gpu = other._gpu;
    this->_bbox = other._bbox;
    this->_need_nms = other._need_nms;
    this->_template = other._template;
    this->_regression = other._regression;
    this->_timeserie = other._timeserie;
    this->_concurrent_predict = other._concurrent_predict;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                 TMLModel>::~ONNXRuntimeLib()
  {
    clear_mllib(APIData());
  }

  // --------------------------------------------------------------------------
  // Lifecycle Management
  // --------------------------------------------------------------------------

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                      TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("gpu"))
      this->_gpu = ad.get("gpu").get<bool>();
    if (ad.has("gpuid"))
      this->_gpuid = ad.get("gpuid").get<int>();
    if (ad.has("nclasses"))
      this->_nclasses = ad.get("nclasses").get<int>();
    if (ad.has("bbox"))
      this->_bbox = ad.get("bbox").get<bool>();
    if (ad.has("max_batch_size"))
      this->_max_batch_size = ad.get("max_batch_size").get<int>();
    else if (ad.has("maxBatchSize"))
      this->_max_batch_size = ad.get("maxBatchSize").get<int>();
    if (ad.has("concurrent_predict"))
      this->_concurrent_predict = ad.get("concurrent_predict").get<bool>();
    if (ad.has("template"))
      {
        this->_template = ad.get("template").get<std::string>();
        this->_logger->info("Model template is {}", this->_template);
        if (this->_template == "yolox")
          {
            this->_mltype = "detection";
            this->_bbox = true;
            this->_need_nms = true;
          }
        else
          {
            throw MLLibBadParamException("Unknown template "
                                         + this->_template);
          }
      }

    if (this->_mlmodel._model.empty())
      {
        this->_logger->error("No ONNX model file found in repository {}",
                             this->_mlmodel._repo);
        throw MLLibBadParamException("No ONNX model file found in repository");
      }

    OrtLoggingLevel log_level
        = ORT_LOGGING_LEVEL_WARNING; // Default to WARNING
    if (ad.has("ort_log_level"))
      {
        std::string level_str = ad.get("ort_log_level").get<std::string>();
        if (level_str == "verbose")
          log_level = ORT_LOGGING_LEVEL_VERBOSE;
        else if (level_str == "info")
          log_level = ORT_LOGGING_LEVEL_INFO;
        else if (level_str == "warning")
          log_level = ORT_LOGGING_LEVEL_WARNING;
        else if (level_str == "error")
          log_level = ORT_LOGGING_LEVEL_ERROR;
        else if (level_str == "fatal")
          log_level = ORT_LOGGING_LEVEL_FATAL;
        else
          this->_logger->warn(
              "Unknown ort_log_level '{}'. Defaulting to warning.", level_str);
      }
    _env_ptr = std::make_unique<Ort::Env>(log_level, "DeepDetect_ORT");

    _session_options = Ort::SessionOptions();
    // Default to 1 for GPU safety, but allow CPU-only users to scale up
    int intra_op_threads = 1;
    if (ad.has("intra_op_threads"))
      {
        intra_op_threads = ad.get("intra_op_threads").get<int>();
        if (intra_op_threads == 0)
          {
            this->_logger->info("intra_op_threads set to 0. ONNX Runtime will "
                                "auto-detect core count");
          }
        else if (intra_op_threads > 128)
          {
            this->_logger->warn("intra_op_threads {} is excessively high",
                                intra_op_threads);
          }
        else if (intra_op_threads < 0)
          {
            this->_logger->error("intra_op_threads cannot be negative");
            throw MLLibBadParamException(
                "intra_op_threads cannot be negative");
          }
      }
    _session_options.SetIntraOpNumThreads(intra_op_threads);
    this->_logger->info("ONNX Runtime: Set intra_op_num_threads to {}",
                        intra_op_threads);

    // ONNX Runtime defaults to true, which maximizes speed but burns CPU
    // cycles busy-waiting
    bool allow_spinning = true;

    if (ad.has("allow_spinning"))
      {
        allow_spinning = ad.get("allow_spinning").get<bool>();
      }

    if (!allow_spinning)
      {
        // Disable busy-waiting for both thread pools
        _session_options.AddConfigEntry("session.intra_op.allow_spinning",
                                        "0");
        _session_options.AddConfigEntry("session.inter_op.allow_spinning",
                                        "0");
        this->_logger->info(
            "ONNX Runtime: Disabled CPU thread spinning (busy-waiting)");
      }
    else
      {
        this->_logger->info(
            "ONNX Runtime: CPU thread spinning enabled (default)");
      }

    _session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    configure_execution_provider(ad);

    try
      {
        this->_logger->info("Loading ONNX model from: {}",
                            this->_mlmodel._model);
        _session = std::make_unique<Ort::Session>(
            *_env_ptr, this->_mlmodel._model.c_str(), _session_options);
      }
    catch (const Ort::Exception &e)
      {
        this->_logger->error("Failed to initialize ONNX Runtime session: {}",
                             e.what());
        throw MLLibInternalException(e.what());
      }

    inspect_model_io();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                 TMLModel>::configure_execution_provider(const APIData &ad)
  {
    (void)ad;
#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    if (this->_gpu)
      {
        try
          {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = this->_gpuid;
            _session_options.AppendExecutionProvider_CUDA(cuda_options);
            this->_logger->info("ONNX Runtime: CUDA Execution Provider "
                                "enabled on GPU device {}",
                                this->_gpuid);
          }
        catch (const Ort::Exception &e)
          {
            this->_logger->error(
                "ONNX Runtime: Failed to load CUDA provider: {}", e.what());
            throw MLLibInternalException(
                std::string("Failed to load CUDA provider: ") + e.what());
          }
      }
    else
      {
        this->_logger->info("ONNX Runtime: Configured for CPU execution");
      }
#else
    if (this->_gpu)
      {
        this->_logger->error("ONNX Runtime: GPU execution requested but build "
                             "is CPU-only or Apple");
        throw MLLibInternalException(
            "GPU requested but ONNX Runtime build is CPU-only or Apple");
      }
    this->_logger->info(
        "ONNX Runtime: Running on CPU (macOS / CPU-only build)");
#endif
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                      TMLModel>::inspect_model_io()
  {
    _input_node_names.clear();
    _input_node_names_ptr.clear();
    size_t num_inputs = _session->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i)
      {
        auto name_ptr = _session->GetInputNameAllocated(i, _allocator);
        _input_node_names.push_back(name_ptr.get());

        // Auto-detect dimensions from first input node
        if (i == 0)
          {
            Ort::TypeInfo type_info = _session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_dims = tensor_info.GetShape();

            if (input_dims.size() == 4)
              {
                if (this->_height <= 0 && input_dims[2] > 0)
                  this->_height = input_dims[2];

                if (this->_width <= 0 && input_dims[3] > 0)
                  this->_width = input_dims[3];
              }
          }
      }
    for (const auto &name : _input_node_names)
      _input_node_names_ptr.push_back(name.c_str());

    _output_node_names.clear();
    _output_node_names_ptr.clear();
    size_t num_outputs = _session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i)
      {
        auto name_ptr = _session->GetOutputNameAllocated(i, _allocator);
        _output_node_names.push_back(name_ptr.get());
      }
    for (const auto &name : _output_node_names)
      _output_node_names_ptr.push_back(name.c_str());

    for (size_t i = 0; i < _input_node_names.size(); ++i)
      this->_logger->info("Discovered input node {}: {}", i,
                          _input_node_names[i]);
    for (size_t i = 0; i < _output_node_names.size(); ++i)
      this->_logger->info("Discovered output node {}: {}", i,
                          _output_node_names[i]);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                      TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    _session.reset();
    _input_node_names.clear();
    _input_node_names_ptr.clear();
    _output_node_names.clear();
    _output_node_names_ptr.clear();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                     TMLModel>::train(const APIData &ad, APIData &out)
  {
    (void)ad;
    (void)out;
    this->_logger->error(
        "Training is not supported by the ONNX Runtime backend");
    return 1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                      TMLModel>::model_type(const std::string &param_file,
                                            std::string &mltype)
  {
    (void)param_file;
    mltype = "onnxruntime";
  }

  // --------------------------------------------------------------------------
  // Inference Pipeline
  // --------------------------------------------------------------------------

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  oatpp::Object<DTO::PredictBody>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                 TMLModel>::predict(const APIData &ad)
  {
    std::unique_lock<std::mutex> lock(_net_mutex, std::defer_lock);
    if (!this->_concurrent_predict)
      lock.lock();

    // Extract DTO from APIData parameters
    oatpp::Object<DTO::ServicePredict> predict_dto;
    if (ad.has("dto"))
      {
        auto any = ad.get("dto").get<oatpp::Any>();
        predict_dto = oatpp::Object<DTO::ServicePredict>(
            std::static_pointer_cast<typename DTO::ServicePredict>(any->ptr));
      }
    else
      {
        predict_dto = ad.createSharedDTO<DTO::ServicePredict>();
      }

    std::string extract_layer = "";
    if (predict_dto->parameters->mllib
        && predict_dto->parameters->mllib->extract_layer)
      {
        extract_layer = predict_dto->parameters->mllib->extract_layer->c_str();
      }

    bool is_bbox = this->_bbox
                   || (predict_dto->parameters->output
                       && predict_dto->parameters->output->bbox);

    std::vector<APIData> vrad;
    std::vector<UnsupervisedResult> unsup_results;

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    struct CudaPredictGuard
    {
      cudaStream_t stream = nullptr;
      bool has_stream = false;
      void *buf = nullptr;

      ~CudaPredictGuard()
      {
#ifdef USE_CUDA_CV
        if (buf)
          cudaFree(buf);
#endif
        if (has_stream)
          cudaStreamDestroy(stream);
      }
    } cuda_guard;
#ifdef USE_CUDA_CV
    std::unique_ptr<cv::cuda::Stream> cv_stream;
#endif
#endif

    TInputConnectorStrategy inputc(this->_inputc);

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    if (this->_gpu)
      {
        if (cudaSetDevice(this->_gpuid) != cudaSuccess)
          throw MLLibInternalException(
              "ONNX Runtime: Failed to set CUDA device");
        if (cudaStreamCreate(&cuda_guard.stream) != cudaSuccess)
          throw MLLibInternalException(
              "ONNX Runtime: Failed to create CUDA stream");
        cuda_guard.has_stream = true;
#ifdef USE_CUDA_CV
        // Dynamically allocate to ensure memory safety if concurrent_predict
        // is true
        if (cudaMalloc(&cuda_guard.buf, this->_max_batch_size
                                            * inputc.channels() * this->_height
                                            * this->_width * sizeof(float))
            != cudaSuccess)
          throw MLLibInternalException(
              "ONNX Runtime: Failed to allocate CUDA input buffer");
        inputc._cuda_buf = static_cast<float *>(cuda_guard.buf);
        // Dynamically initialize only when a GPU is active
        cv_stream = std::make_unique<cv::cuda::Stream>(
            cv::cuda::StreamAccessor::wrapStream(cuda_guard.stream));
        inputc._cuda_stream = cv_stream.get();
#endif
      }
#endif

    this->_stats.transform_start();
    inputc.transform(predict_dto);
    this->_stats.transform_end();
    this->_stats.inc_inference_count(inputc._batch_size);

    if (inputc.width() != this->_width || inputc.height() != this->_height)
      {
        throw MLLibBadParamException(
            "Model height and/or width differ from the input data size. "
            "Input data size should be "
            + std::to_string(this->_width) + "x"
            + std::to_string(this->_height));
      }

    int idoffset = 0;
    int batch_size = 0;
    while ((batch_size = inputc.process_batch(this->_max_batch_size)) > 0)
      {
        std::vector<int64_t> input_shape
            = { static_cast<int64_t>(batch_size),
                static_cast<int64_t>(
                    inputc.channels()), // Using your new channels() method!
                static_cast<int64_t>(this->_height),
                static_cast<int64_t>(this->_width) };
        size_t input_tensor_size
            = batch_size * inputc.channels() * this->_height * this->_width;

        Ort::Value input_tensor{ nullptr };

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
        if (this->_gpu)
          {
            cudaStreamSynchronize(cuda_guard.stream);
          }

#ifdef USE_CUDA_CV
        if (this->_gpu && inputc._cuda)
          {
            Ort::MemoryInfo mem_info("Cuda", OrtDeviceAllocator, this->_gpuid,
                                     OrtMemTypeDefault);
            input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, static_cast<float *>(cuda_guard.buf),
                input_tensor_size, input_shape.data(), input_shape.size());
          }
        else
#endif
#endif
          {
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, inputc.data(), input_tensor_size, input_shape.data(),
                input_shape.size());
          }
        std::vector<Ort::Value> output_tensors;

        try
          {
            std::vector<const char *> run_output_names
                = _output_node_names_ptr;
            if (!extract_layer.empty() && extract_layer != "last")
              {
                run_output_names = { extract_layer.c_str() };
              }
            output_tensors = _session->Run(
                Ort::RunOptions{ nullptr }, _input_node_names_ptr.data(),
                &input_tensor, 1, run_output_names.data(),
                run_output_names.size());
          }
        catch (const Ort::Exception &e)
          {
            std::string err_msg = e.what();
            // Intercept models that crash on zero detections (empty tensors)
            if (err_msg.find("Input shape:{0,") != std::string::npos)
              {
                this->_logger->info(
                    "ONNX Runtime: 0 objects detected (caught empty tensor). "
                    "Returning empty prediction.");
                OutputConnectorConfig conf;
                conf._nclasses = this->_nclasses;
                if (is_bbox)
                  conf._has_bbox = true;

                if (extract_layer.empty())
                  {
                    TOutputConnectorStrategy tout(this->_outputc);
                    std::vector<APIData> empty_vrad;
                    for (int b = 0; b < batch_size; ++b)
                      {
                        APIData rad;
                        rad.add("uri", inputc._ids.at(idoffset + b));
                        rad.add("loss", 0.0);
                        rad.add("probs", std::vector<double>());
                        rad.add("cats", std::vector<std::string>());
                        if (is_bbox)
                          rad.add("bboxes", std::vector<APIData>());
                        empty_vrad.push_back(rad);
                      }
                    tout.add_results(empty_vrad);
                    return tout.finalize(
                        predict_dto->parameters->output, conf,
                        static_cast<MLModel *>(&this->_mlmodel));
                  }
                else
                  {
                    UnsupervisedOutput unsupo;
                    unsupo.set_results(std::vector<UnsupervisedResult>());
                    return unsupo.finalize(
                        predict_dto->parameters->output, conf,
                        static_cast<MLModel *>(&this->_mlmodel));
                  }
              }
            this->_logger->error(
                "Inference execution failed in ONNX Runtime: {}", err_msg);
            throw MLLibInternalException(err_msg);
          }

        if (output_tensors.empty())
          {
            this->_logger->error(
                "ONNX Runtime: Inference returned no output tensors.");
            throw MLLibInternalException(
                "ONNX Runtime returned no output tensors");
          }

        // Handle extract_layer tensor extraction
        if (!extract_layer.empty())
          {
            // If "last" returned multiple outputs, iterate through all and
            // concatenate.
            for (int b = 0; b < batch_size; ++b)
              {
                UnsupervisedResult result;
                result._uri = inputc._ids.at(idoffset + b);
                result._loss = 0.0;

                for (size_t t = 0; t < output_tensors.size(); ++t)
                  {
                    auto tensor_info
                        = output_tensors[t].GetTensorTypeAndShapeInfo();
                    if (tensor_info.GetElementCount() == 0)
                      continue;
                    std::vector<int64_t> out_shape = tensor_info.GetShape();
                    ONNXTensorElementDataType type
                        = tensor_info.GetElementType();

                    size_t elements_per_batch = 1;
                    for (size_t i = 1; i < out_shape.size(); ++i)
                      {
                        elements_per_batch *= out_shape[i];
                      }

                    // Handle standard 32-bit floats
                    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                      {
                        float *out_data
                            = output_tensors[t]
                                  .template GetTensorMutableData<float>();
                        float *batch_start
                            = out_data + (b * elements_per_batch);
                        result._vals.insert(result._vals.end(), batch_start,
                                            batch_start + elements_per_batch);
                      }
                    // Handle 64-bit integers (e.g., class indices)
                    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                      {
                        int64_t *out_data
                            = output_tensors[t]
                                  .template GetTensorMutableData<int64_t>();
                        int64_t *batch_start
                            = out_data + (b * elements_per_batch);
                        for (size_t i = 0; i < elements_per_batch; ++i)
                          {
                            result._vals.push_back(
                                static_cast<double>(batch_start[i]));
                          }
                      }
                    // Fallback for unsupported types
                    else
                      {
                        this->_logger->warn(
                            "ONNX Runtime: extract_layer encountered "
                            "unsupported tensor type {}",
                            static_cast<int>(type));
                      }
                  }
                unsup_results.push_back(std::move(result));
              }
          }
        // Parse detection bounding boxes
        else if (is_bbox)
          {
            float *out_data = output_tensors.front()
                                  .template GetTensorMutableData<float>();
            auto tensor_info
                = output_tensors.front().GetTensorTypeAndShapeInfo();
            std::vector<int64_t> out_shape = tensor_info.GetShape();

            double conf_thresh = 0.0;
            double nms_thresh = 0.45;
            int best_bbox = -1;
            if (predict_dto->parameters->output)
              {
                conf_thresh
                    = predict_dto->parameters->output->confidence_threshold;
                nms_thresh = predict_dto->parameters->output->nms_threshold;
                best_bbox = predict_dto->parameters->output->best_bbox;
              }

            std::vector<float> yolo_out;
            const float *yolo_ptr = nullptr;
            int max_dets = 0;

            if (this->_template == "yolox")
              {
                if (out_shape.size() != 3 || out_shape[2] <= 5)
                  {
                    this->_logger->error(
                        "ONNX Runtime YOLOX output expects shape [batch, "
                        "anchors, 5 + num_classes], got rank {} with last dim "
                        "{}",
                        out_shape.size(),
                        out_shape.empty() ? -1 : out_shape.back());
                    throw MLLibBadParamException(
                        "Invalid YOLOX output tensor shape");
                  }
                if (tensor_info.GetElementCount() > 0)
                  {
                    std::vector<float> raw_out(
                        out_data, out_data + tensor_info.GetElementCount());
                    yolo_out = yolo_utils::parse_yolo_output(
                        raw_out, batch_size, static_cast<int>(out_shape[1]),
                        static_cast<int>(out_shape[2]), this->_nclasses,
                        inputc.width(), inputc.height());
                    yolo_ptr = yolo_out.data();
                  }
              }
            else
              {
                if (out_shape.size() != 3 || out_shape[2] != 7)
                  {
                    this->_logger->error(
                        "ONNX Runtime bbox output expects shape [batch, "
                        "max_detections, 7], got rank {} with last dim {}",
                        out_shape.size(),
                        out_shape.empty() ? -1 : out_shape.back());
                    throw MLLibBadParamException(
                        "ONNX Runtime bbox model output must have shape "
                        "[batch, "
                        "max_detections, 7]");
                  }

                // Expecting [batch, max_detections, 7] layout
                if (tensor_info.GetElementCount() > 0)
                  max_dets = (out_shape.size() >= 2) ? out_shape[1] : 100;
              }

            for (int b = 0; b < batch_size; ++b)
              {
                APIData rad;
                std::string img_id = inputc._ids.at(idoffset + b);
                rad.add("uri", img_id);
                rad.add("loss", 0.0);

                std::vector<double> probs;
                std::vector<std::string> cats;
                std::vector<APIData> bboxes;

                int orig_w = this->_width;
                int orig_h = this->_height;

                if (inputc._imgs_size.find(img_id) != inputc._imgs_size.end())
                  {
                    orig_h = inputc._imgs_size[img_id].first;
                    orig_w = inputc._imgs_size[img_id].second;
                  }

                auto push_detection = [&](const float *det) {
                  int label_id = static_cast<int>(det[1]);
                  float score = det[2];
                  if (score < conf_thresh)
                    return false;
                  float xmin = std::max(det[3], 0.0f) * orig_w;
                  float ymin = std::max(det[4], 0.0f) * orig_h;
                  float xmax = std::min(det[5], 1.0f) * orig_w;
                  float ymax = std::min(det[6], 1.0f) * orig_h;

                  probs.push_back(static_cast<double>(score));
                  cats.push_back(this->_mlmodel.get_hcorresp(label_id));

                  APIData ad_bbox;
                  ad_bbox.add("xmin", static_cast<double>(xmin));
                  ad_bbox.add("ymin", static_cast<double>(ymin));
                  ad_bbox.add("xmax", static_cast<double>(xmax));
                  ad_bbox.add("ymax", static_cast<double>(ymax));
                  bboxes.push_back(ad_bbox);
                  return true;
                };

                if (this->_template == "yolox")
                  {
                    int num_anchors = static_cast<int>(out_shape[1]);
                    if (yolo_ptr != nullptr)
                      {
                        const float *batch_yolo
                            = yolo_ptr + (b * num_anchors * 7);
                        for (int d = 0; d < num_anchors; ++d)
                          {
                            const float *det = batch_yolo + (d * 7);
                            // Sorted by descending score in parse_yolo_output
                            if (!push_detection(det))
                              break;
                          }
                      }
                  }
                else
                  {
                    for (int d = 0; d < max_dets; ++d)
                      {
                        if (!this->_need_nms && best_bbox > 0
                            && bboxes.size() >= static_cast<size_t>(best_bbox))
                          break;
                        const float *det = out_data + ((b * max_dets + d) * 7);
                        if (det[0] < 0.0f)
                          break;
                        push_detection(det);
                      }
                  }
                if (this->_need_nms)
                  {
                    bbox_utils::nms_sorted_bboxes(bboxes, probs, cats,
                                                  nms_thresh, best_bbox);
                  }

                rad.add("probs", probs);
                rad.add("cats", cats);
                rad.add("bboxes", bboxes);
                vrad.push_back(rad);
              }
          }
        // Parse standard classification
        else
          {
            float *out_data = output_tensors.front()
                                  .template GetTensorMutableData<float>();
            auto tensor_info
                = output_tensors.front().GetTensorTypeAndShapeInfo();
            std::vector<int64_t> out_shape = tensor_info.GetShape();
            double conf_thresh = 0.0;
            if (predict_dto->parameters->output)
              {
                conf_thresh
                    = predict_dto->parameters->output->confidence_threshold;
              }
            size_t elements_per_batch = 1;
            if (tensor_info.GetElementCount() > 0)
              {
                for (size_t i = 1; i < out_shape.size(); ++i)
                  {
                    elements_per_batch *= out_shape[i];
                  }
              }

            // Prevent out-of-bounds if the model returns fewer outputs than
            // nclasses
            int num_classes = std::min(static_cast<size_t>(this->_nclasses),
                                       elements_per_batch);

            for (int b = 0; b < batch_size; ++b)
              {
                APIData rad;
                std::string img_id = inputc._ids.at(idoffset + b);
                rad.add("uri", img_id);
                rad.add("loss", 0.0);

                std::vector<double> probs;
                std::vector<std::string> cats;

                if (tensor_info.GetElementCount() > 0)
                  {
                    float *batch_start = out_data + (b * elements_per_batch);
                    for (int c = 0; c < num_classes; ++c)
                      {
                        double prob = static_cast<double>(batch_start[c]);
                        if (prob < conf_thresh)
                          continue;
                        probs.push_back(prob);
                        cats.push_back(this->_mlmodel.get_hcorresp(c));
                      }
                  }

                rad.add("probs", probs);
                rad.add("cats", cats);
                vrad.push_back(rad);
              }
          }
        idoffset += batch_size;
      }
    // If the loop never ran (0 valid images), return empty
    if (vrad.empty() && unsup_results.empty())
      {
        return oatpp::Object<DTO::PredictBody>();
      }

    // Pack into DTO response
    OutputConnectorConfig conf;
    conf._nclasses = this->_nclasses;
    if (is_bbox)
      conf._has_bbox = true;

    if (extract_layer.empty())
      {
        TOutputConnectorStrategy tout(this->_outputc);
        tout.add_results(vrad);
        return tout.finalize(predict_dto->parameters->output, conf,
                             static_cast<MLModel *>(&this->_mlmodel));
      }
    else
      {
        UnsupervisedOutput unsupo;
        unsupo.set_results(std::move(unsup_results));
        return unsupo.finalize(predict_dto->parameters->output, conf,
                               static_cast<MLModel *>(&this->_mlmodel));
      }
  }

  // --------------------------------------------------------------------------
  // Explicit Template Instantiations
  // --------------------------------------------------------------------------
  template class ONNXRuntimeLib<ImgONNXRuntimeInputFileConn, SupervisedOutput,
                                ONNXRuntimeModel>;

} // namespace dd
