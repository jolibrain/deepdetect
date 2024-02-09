/* tensorrtlib.h ---  */

/* Copyright (C) 2019 Jolibrain http://www.jolibrain.com */

/* Author: Guillaume Infantes <guillaume.infantes@jolibrain.com> */

/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 3 */
/* of the License, or (at your option) any later version. */

/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */

/* You should have received a copy of the GNU General Public License */
/* along with this program. If not, see <http://www.gnu.org/licenses/>. */

#ifndef TENSORRTLIB_H
#define TENSORRTLIB_H

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include "apidata.h"
#include "tensorrtmodel.h"
#include "error_recorder.hpp"

namespace dd
{

  struct TRTInferDeleter
  {
    template <typename T> void operator()(T *obj) const
    {
      if (obj)
        {
          obj->destroy();
        }
    }
  };

  template <typename T>
  using TRTUniquePtr = std::unique_ptr<T, TRTInferDeleter>;

  class TRTLogger : public nvinfer1::ILogger
  {
  public:
    TRTLogger(nvinfer1::ILogger::Severity severity
              = nvinfer1::ILogger::Severity::kWARNING)
        : mReportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity,
             const char *msg) noexcept override
    {
      switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        case nvinfer1::ILogger::Severity::kERROR:
          this->_logger->error(msg);
          break;
        case nvinfer1::ILogger::Severity::kWARNING:
          this->_logger->warn(msg);
          break;
        case nvinfer1::ILogger::Severity::kINFO:
        default:
          this->_logger->info(msg);
          break;
        }
    }

    void setReportableSeverity(nvinfer1::ILogger::Severity severity)
    {
      mReportableSeverity = severity;
    }

    void setLogger(std::shared_ptr<spdlog::logger> &l)
    {
      _logger = l.get();
    }

  private:
    nvinfer1::ILogger::Severity mReportableSeverity;
    spdlog::logger *_logger;
  };

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = TensorRTModel>
  class TensorRTLib : public MLLib<TInputConnectorStrategy,
                                   TOutputConnectorStrategy, TMLModel>
  {

  public:
    TensorRTLib(const TensorRTModel &tmodel);
    TensorRTLib(TensorRTLib &&tl) noexcept;
    ~TensorRTLib();

    /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &ad);

    int train(const APIData &ad, APIData &out);

    oatpp::Object<DTO::PredictBody> predict(const APIData &ad);

    void model_type(const std::string &param_file, std::string &mltype);

  public:
    int _nclasses = 0;
    int _width = 0;  /**< default model width. */
    int _height = 0; /**< default model height. */
    int _dla = -1;
    nvinfer1::DataType _datatype = nvinfer1::DataType::kFLOAT;
    int _max_batch_size = 48;
    size_t _max_workspace_size = 1 << 30; // 1GB
    int _results_height = -1;
    std::string _engineFileName = "TRTengine";
    bool _readEngine = true;
    bool _writeEngine = true;
    std::string _arch;
    int _gpuid = 0;
    std::string
        _template; /**< template for models that require specific treatment */

    //!< The TensorRT engine used to run the network
    std::shared_ptr<TRTErrorRecorder> _error_recorder = nullptr;
    std::shared_ptr<nvinfer1::IInt8Calibrator> _calibrator = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine = nullptr;
    std::shared_ptr<nvinfer1::IBuilder> _builder = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> _context = nullptr;
    std::shared_ptr<nvinfer1::IBuilderConfig> _builderc = nullptr;
    std::shared_ptr<nvinfer1::IRuntime> _runtime = nullptr;

    bool _bbox = false;
    bool _ctc = false;
    bool _regression = false;
    bool _timeserie = false;

    // detection
    bool _need_nms = false;

    // XXX: Single buffer vector kept for backward compatibility
    std::vector<void *> _buffers;

    bool _TRTContextReady = false;

    std::vector<std::string> _inputNames;
    std::vector<int> _inputIndices;
    std::vector<std::string> _outputNames;
    std::vector<int> _outputIndices;

    bool _first_predict
        = true; // do some cuda allocations only at first predict

    bool _explicit_batch
        = false; /**< whether TRT uses explicit batch model (ONNX). */

    std::vector<float> _floatOut;
    std::vector<int> _keepCount;

    nvinfer1::Dims _dims;

    std::mutex
        _net_mutex; /**< mutex around net, e.g. no concurrent predict calls as
                       net is not re-instantiated. Use batches instead. */

    nvinfer1::ICudaEngine *read_engine_from_caffe(const std::string &out_blob);

    nvinfer1::ICudaEngine *read_engine_from_onnx();

    /** create buffer array with inputs and output count */
    void allocate_buffer_array();
  };
}
#endif
