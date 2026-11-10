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

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::ONNXRuntimeLib(
      const ONNXRuntimeModel &model)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, ONNXRuntimeModel>(model),
        _env(ORT_LOGGING_LEVEL_WARNING, "DeepDetect_ORT")
  {
    this->_libname = "onnxruntime";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::ONNXRuntimeLib(
      ONNXRuntimeLib &&other) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, ONNXRuntimeModel>(std::move(other)),
        _env(std::move(other._env)),
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
    this->_template = other._template;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~ONNXRuntimeLib()
  {
    clear_mllib(APIData());
  }

  // --------------------------------------------------------------------------
  // Lifecycle Management
  // --------------------------------------------------------------------------

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(
      const APIData &ad)
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
    if (ad.has("concurrent_predict"))
      this->_concurrent_predict = ad.get("concurrent_predict").get<bool>();

    if (this->_mlmodel._model.empty())
      {
        this->_logger->error("No ONNX model file found in repository {}", this->_mlmodel._repo);
        throw MLLibBadParamException("No ONNX model file found in repository");
      }

    _session_options = Ort::SessionOptions();
    _session_options.SetIntraOpNumThreads(1);
    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    configure_execution_provider(ad);

    try
      {
        this->_logger->info("Loading ONNX model from: {}", this->_mlmodel._model);
        _session = std::make_unique<Ort::Session>(_env, this->_mlmodel._model.c_str(), _session_options);
      }
    catch (const Ort::Exception &e)
      {
        this->_logger->error("Failed to initialize ONNX Runtime session: {}", e.what());
        throw MLLibInternalException(e.what());
      }

    inspect_model_io();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::configure_execution_provider(
      const APIData &ad)
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
            this->_logger->info("ONNX Runtime: CUDA Execution Provider enabled on GPU device {}", this->_gpuid);
          }
        catch (const Ort::Exception &e)
          {
            this->_logger->error("ONNX Runtime: Failed to load CUDA provider: {}", e.what());
            throw MLLibInternalException(std::string("Failed to load CUDA provider: ") + e.what());
          }
      }
    else
      {
        this->_logger->info("ONNX Runtime: Configured for CPU execution");
      }
#else
    if (this->_gpu)
      {
        this->_logger->error("ONNX Runtime: GPU execution requested but build is CPU-only or Apple");
        throw MLLibInternalException("GPU requested but ONNX Runtime build is CPU-only or Apple");
      }
    this->_logger->info("ONNX Runtime: Running on CPU (macOS / CPU-only build)");
#endif
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::inspect_model_io()
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
                if (this->_mlmodel._height <= 0 && input_dims[2] > 0)
                  this->_mlmodel._height = input_dims[2];

                if (this->_mlmodel._width <= 0 && input_dims[3] > 0)
                  this->_mlmodel._width = input_dims[3];
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
      this->_logger->info("Discovered input node {}: {}", i, _input_node_names[i]);
    for (size_t i = 0; i < _output_node_names.size(); ++i)
      this->_logger->info("Discovered output node {}: {}", i, _output_node_names[i]);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(
      const APIData &ad)
  {
    (void)ad;
    _session.reset();
    _input_node_names.clear();
    _input_node_names_ptr.clear();
    _output_node_names.clear();
    _output_node_names_ptr.clear();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(
      const APIData &ad, APIData &out)
  {
    (void)ad;
    (void)out;
    this->_logger->error("Training is not supported by the ONNX Runtime backend");
    return 1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::model_type(
      const std::string &param_file, std::string &mltype)
  {
    (void)param_file;
    mltype = "onnxruntime";
  }

  // --------------------------------------------------------------------------
  // Inference Pipeline
  // --------------------------------------------------------------------------

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  oatpp::Object<DTO::PredictBody>
  ONNXRuntimeLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::predict(
      const APIData &ad)
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

    TInputConnectorStrategy inputc(this->_inputc);

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    cudaStream_t cstream;
    void *cuda_input_buf = nullptr;

    if (this->_gpu)
      {
        cudaStreamCreate(&cstream);
#ifdef USE_CUDA_CV
        // Dynamically allocate to ensure memory safety if concurrent_predict is true
        cudaMalloc(&cuda_input_buf, this->_max_batch_size * 3 * this->_height * this->_width * sizeof(float));
        inputc._cuda_buf = static_cast<float *>(cuda_input_buf);
        auto cv_stream = cv::cuda::StreamAccessor::wrapStream(cstream);
        inputc._cuda_stream = &cv_stream;
#endif
      }
#endif

    this->_stats.transform_start();
    try
      {
        inputc.transform(predict_dto);
      }
    catch (...)
      {
#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
        if (this->_gpu)
          {
#ifdef USE_CUDA_CV
            if (cuda_input_buf) cudaFree(cuda_input_buf);
#endif
            cudaStreamDestroy(cstream);
          }
#endif
        throw;
      }
    this->_stats.transform_end();

    if (inputc.width() != this->_mlmodel._width || inputc.height() != this->_mlmodel._height)
      {
#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
        if (this->_gpu)
          {
#ifdef USE_CUDA_CV
            if (cuda_input_buf) cudaFree(cuda_input_buf);
#endif
            cudaStreamDestroy(cstream);
          }
#endif
        throw MLLibBadParamException(
            "Model height and/or width differ from the input data size. "
            "Input data size should be " + std::to_string(this->_mlmodel._width) + "x" + std::to_string(this->_mlmodel._height));
      }

    int batch_size = inputc.process_batch(this->_max_batch_size);
    if (batch_size <= 0)
      {
#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
        if (this->_gpu)
          {
#ifdef USE_CUDA_CV
            if (cuda_input_buf) cudaFree(cuda_input_buf);
#endif
            cudaStreamDestroy(cstream);
          }
#endif
        return oatpp::Object<DTO::PredictBody>();
      }

    std::vector<int64_t> input_shape = {
      static_cast<int64_t>(batch_size),
      3,
      static_cast<int64_t>(this->_height),
      static_cast<int64_t>(this->_width)
    };
    size_t input_tensor_size = batch_size * 3 * this->_height * this->_width;

    Ort::Value input_tensor{nullptr};

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    if (this->_gpu)
      {
        cudaStreamSynchronize(cstream);
      }

#ifdef USE_CUDA_CV
    if (this->_gpu && inputc._cuda)
      {
        Ort::MemoryInfo mem_info("Cuda", OrtDeviceAllocator, this->_gpuid, OrtMemTypeDefault);
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            static_cast<float *>(cuda_input_buf),
            input_tensor_size,
            input_shape.data(),
            input_shape.size());
      }
    else
#endif
#endif
      {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            inputc.data(),
            input_tensor_size,
            input_shape.data(),
            input_shape.size());
      }

    std::vector<Ort::Value> output_tensors;
    try
      {
        output_tensors = _session->Run(
            Ort::RunOptions{nullptr},
            _input_node_names_ptr.data(),
            &input_tensor,
            1,
            _output_node_names_ptr.data(),
            _output_node_names_ptr.size());
      }
    catch (const Ort::Exception &e)
      {
#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
        if (this->_gpu)
          {
#ifdef USE_CUDA_CV
            if (cuda_input_buf) cudaFree(cuda_input_buf);
#endif
            cudaStreamDestroy(cstream);
          }
#endif
        this->_logger->error("Inference execution failed in ONNX Runtime: {}", e.what());
        throw MLLibInternalException(e.what());
      }

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
    if (this->_gpu)
      {
#ifdef USE_CUDA_CV
        if (cuda_input_buf) cudaFree(cuda_input_buf);
#endif
        cudaStreamDestroy(cstream);
      }
#endif

    // Prepare DeepDetect output collections
    TOutputConnectorStrategy tout(this->_outputc);
    std::vector<APIData> vrad;

    // Parse detection bounding boxes
    if (this->_bbox)
      {
        float *out_data = output_tensors.front().template GetTensorMutableData<float>();
        auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = tensor_info.GetShape();

        // Expecting [batch, max_detections, 7] layout
        int max_dets = (out_shape.size() >= 2) ? out_shape[1] : 100;

        for (int b = 0; b < batch_size; ++b)
          {
            APIData rad;
            std::string img_id = inputc._ids.at(b);
            rad.add("uri", img_id);
            rad.add("loss", 0.0);

            std::vector<double> probs;
            std::vector<std::string> cats;
            std::vector<APIData> bboxes;

            int orig_w = this->_width;
            int orig_h = this->_height;

            if (inputc._imgs_size.find(img_id) != inputc._imgs_size.end())
              {
                orig_w = inputc._imgs_size[img_id].first;
                orig_h = inputc._imgs_size[img_id].second;
              }

            for (int d = 0; d < max_dets; ++d)
              {
                int base_idx = (b * max_dets + d) * 7;
                float batch_idx = out_data[base_idx + 0];

                // Padding indicator from TRTDetectionWrapper
                if (batch_idx < 0.0f)
                  break;

                int label_id = static_cast<int>(out_data[base_idx + 1]);
                float score = out_data[base_idx + 2];
                float xmin = out_data[base_idx + 3] * orig_w;
                float ymin = out_data[base_idx + 4] * orig_h;
                float xmax = out_data[base_idx + 5] * orig_w;
                float ymax = out_data[base_idx + 6] * orig_h;

                probs.push_back(static_cast<double>(score));
                cats.push_back(this->_mlmodel.get_hcorresp(label_id));

                APIData ad_bbox;
                ad_bbox.add("xmin", static_cast<double>(xmin));
                ad_bbox.add("ymin", static_cast<double>(ymin));
                ad_bbox.add("xmax", static_cast<double>(xmax));
                ad_bbox.add("ymax", static_cast<double>(ymax));
                bboxes.push_back(ad_bbox);
              }

            rad.add("probs", probs);
            rad.add("cats", cats);
            rad.add("bboxes", bboxes);
            vrad.push_back(rad);
          }
      }

    // Pack into DTO response
    OutputConnectorConfig conf;
    conf._nclasses = this->_nclasses;
    if (this->_bbox)
      conf._has_bbox = true;

    tout.add_results(vrad);
    return tout.finalize(predict_dto->parameters->output, conf,
                         static_cast<MLModel *>(&this->_mlmodel));
  }

  // --------------------------------------------------------------------------
  // Explicit Template Instantiations
  // --------------------------------------------------------------------------
  template class ONNXRuntimeLib<ImgONNXRuntimeInputFileConn, SupervisedOutput, ONNXRuntimeModel>;

} // namespace dd