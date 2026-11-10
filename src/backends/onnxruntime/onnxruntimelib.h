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

#ifndef ONNXRUNTIMELIB_H
#define ONNXRUNTIMELIB_H

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "apidata.h"
#include "mllibstrategy.h"
#include "onnxruntimemodel.h"
#include "onnxruntimeinputconns.h"

namespace dd
{
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = ONNXRuntimeModel>
  class ONNXRuntimeLib : public MLLib<TInputConnectorStrategy,
                                      TOutputConnectorStrategy, TMLModel>
  {
  public:
    ONNXRuntimeLib(const ONNXRuntimeModel &model);
    ONNXRuntimeLib(ONNXRuntimeLib &&other) noexcept;
    virtual ~ONNXRuntimeLib();

    /*- Lifecycle methods from MLLib -*/
    void init_mllib(const APIData &ad);
    void clear_mllib(const APIData &ad);
    int train(const APIData &ad, APIData &out);
    oatpp::Object<DTO::PredictBody> predict(const APIData &ad);
    void model_type(const std::string &param_file, std::string &mltype);

  public:
    // Model and image dimensions
    int _nclasses = 0;
    int _width = 0;
    int _height = 0;
    int _max_batch_size = 1;

    // Hardware and Execution Provider options
    int _gpuid = 0;
    bool _gpu = false;

    // Task configuration
    bool _bbox = false;
    bool _regression = false;
    bool _timeserie = false;
    bool _concurrent_predict = false;
    std::string _template;

    // ONNX Runtime environment and session handles
    Ort::Env _env;
    Ort::SessionOptions _session_options;
    std::unique_ptr<Ort::Session> _session = nullptr;
    Ort::AllocatorWithDefaultOptions _allocator;

    // Model I/O node names
    std::vector<std::string> _input_node_names;
    std::vector<std::string> _output_node_names;
    std::vector<const char *> _input_node_names_ptr;
    std::vector<const char *> _output_node_names_ptr;

    // Detected input/output shapes
    std::vector<int64_t> _input_dims;
    std::vector<int64_t> _output_dims;

    // Inference serialization guard
    std::mutex _net_mutex;

  private:
    void configure_execution_provider(const APIData &ad);
    void inspect_model_io();
  };
}

#endif // ONNXRUNTIMELIB_H