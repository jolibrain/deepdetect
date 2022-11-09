/**
 * DeepDetect
 * Copyright (c) 2022 Jolibrain
 * Authors: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DD_TRT_ERROR_RECORDER
#define DD_TRT_ERROR_RECORDER

#include <vector>
#include <string>
#include <mutex>
#include <exception>
#include <atomic>

#include <NvInferRuntimeCommon.h>

namespace dd
{

  struct Error
  {
    nvinfer1::ErrorCode _code;
    std::string _desc;
  };

  class TRTErrorRecorder : public nvinfer1::IErrorRecorder
  {
  public:
    TRTErrorRecorder(std::shared_ptr<spdlog::logger> logger) : _logger(logger)
    {
    }

    ~TRTErrorRecorder() noexcept override
    {
    }

    int32_t getNbErrors() const noexcept override
    {
      return static_cast<int32_t>(_errors.size());
    }

    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
    {
      return _errors.at(errorIdx)._code;
    }

    nvinfer1::IErrorRecorder::ErrorDesc
    getErrorDesc(int32_t errorIdx) const noexcept override
    {
      return _errors.at(errorIdx)._desc.c_str();
    }

    bool hasOverflowed() const noexcept override
    {
      return false;
    }

    void clear() noexcept override
    {
      try
        {
          std::lock_guard<std::mutex> guard(_errors_mtx);
          _errors.clear();
        }
      catch (std::exception &e)
        {
          _logger->error("TRTErroRecorder::clear error: {}", e.what());
        }
    }

    // API used by TensorRT to report Error information to the application.
    bool
    reportError(nvinfer1::ErrorCode val,
                nvinfer1::IErrorRecorder::ErrorDesc desc) noexcept override
    {
      try
        {
          std::lock_guard<std::mutex> guard(_errors_mtx);
          _errors.push_back(Error{ val, std::string(desc) });
          _logger->error("TRT Error code={}: {}", static_cast<int32_t>(val),
                         std::string(desc));
        }
      catch (std::exception &e)
        {
          _logger->error("TRTErroRecorder::reportError error: {}", e.what());
        }
      return true;
    }

    RefCount incRefCount() noexcept override
    {
      return ++_ref_count;
    }

    RefCount decRefCount() noexcept override
    {
      return --_ref_count;
    }

  private:
    std::vector<Error> _errors;
    std::shared_ptr<spdlog::logger> _logger; /**< dd logger */

    // Mutex to hold when locking mErrorStack.
    std::mutex _errors_mtx;

    std::atomic<int32_t> _ref_count{ 0 };
  };
}

#endif // DD_TRT_ERROR_RECORDER
