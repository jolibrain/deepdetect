/**
 * DeepDetect
 * Copyright (c) 2023 Jolibrain
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef TENSORRTCALIBRATOR_H
#define TENSORRTCALIBRATOR_H

#include <fstream>

#include "mllibstrategy.h"
#include "NvInfer.h"

namespace dd
{
  template <typename TConnector>
  class TRTCalibrator : public nvinfer1::IInt8EntropyCalibrator2
  {
  public:
    TRTCalibrator(TConnector *connector, const std::string &model_repo,
                  int max_batch_size, bool use_cache,
                  std::shared_ptr<spdlog::logger> logger)
        : _conn{ connector }, _logger(logger), _max_batch_size(max_batch_size),
          _use_cache(use_cache),
          _calibration_table_path(model_repo + "/calibration_table")
    {
      if (!use_cache)
        {
          // XXX(louis): works only for images
          if (_conn->_bw)
            _input_size = _conn->_height * _conn->_width;
          else
            _input_size = _conn->_height * _conn->_width * 3;

          auto result = cudaMalloc(&_input_buf, _max_batch_size * _input_size
                                                    * sizeof(float));

          if (result)
            throw MLLibInternalException(
                "Could not allocate input buffer for "
                "model calibration (size="
                + std::to_string(_input_size * _max_batch_size) + ")");
#ifdef USE_CUDA_CV
          _conn->_cuda_buf = static_cast<float *>(_input_buf);
#endif
        }
    }

    virtual ~TRTCalibrator()
    {
      cudaFree(_input_buf);
    }

    int getBatchSize() const noexcept override
    {
      return _max_batch_size;
    }

    /** Returns next batch from input connector */
    bool getBatch(void *bindings[], const char *names[],
                  int nbBindings) noexcept override
    {
      // only one binding
      (void)names;
      (void)nbBindings;

      if (_use_cache)
        return false;

      int num_processed = _conn->process_batch(_max_batch_size);
      if (num_processed == 0)
        return false;
#ifdef USE_CUDA_CV
      if (!_conn->_cuda)
#endif
        {
          bool result
              = cudaMemcpyAsync(_input_buf, _conn->data(),
                                num_processed * _input_size * sizeof(float),
                                cudaMemcpyHostToDevice);
          if (result)
            return false;
        }
      bindings[0] = _input_buf;
      return true;
    }

    /** read calibration table from disk */
    const void *readCalibrationCache(size_t &length) noexcept override
    {
      if (!_use_cache)
        return nullptr;

      _calibration_cache.clear();
      _logger->info("reading cache at {}", _calibration_table_path);
      std::ifstream input(_calibration_table_path, std::ios::binary);
      input >> std::noskipws;
      if (input.good())
        {
          // TODO logger
          std::copy(std::istream_iterator<char>(input),
                    std::istream_iterator<char>(),
                    std::back_inserter(_calibration_cache));
        }
      else
        {
          _logger->error(
              "No int8 calibration data found, please run a calibration "
              "inference with mllib.calibration = true");
        }
      length = _calibration_cache.size();
      return _calibration_cache.data();
    }

    /** write calibration table to disk */
    void writeCalibrationCache(const void *cache,
                               size_t length) noexcept override
    {
      std::ofstream output(_calibration_table_path, std::ios::binary);
      output.write(reinterpret_cast<const char *>(cache), length);
    }

  private:
    TConnector *_conn;
    std::shared_ptr<spdlog::logger> _logger;

    int _max_batch_size;
    /// input size (not batched)
    int _input_size;
    void *_input_buf = nullptr;
    bool _use_cache{ true };
    std::string _calibration_table_path;
    std::vector<char> _calibration_cache;
  };
}

#endif
