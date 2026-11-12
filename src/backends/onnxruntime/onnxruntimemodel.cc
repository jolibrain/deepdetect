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

#include "onnxruntimemodel.h"
#include <utils/fileops.hpp>

namespace dd
{
  int ONNXRuntimeModel::read_from_repository(
      const std::shared_ptr<spdlog::logger> &logger)
  {
    static const std::string onnx_ext = ".onnx";
    static const std::string corresp_name = "corresp";

    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(_repo, true, false, false, lfiles);
    if (e != 0)
      {
        logger->error("error reading or listing models in repository {}",
                      _repo);
        return 1;
      }

    std::string modelf;
    std::string correspf;
    long int model_t = -1;

    for (const auto &file : lfiles)
      {
        // Pick the most recently modified .onnx model if multiple exist
        // Make sure we don't try to pick up any .onnx.data files instead
        if (file.size() >= onnx_ext.size()
            && file.compare(file.size() - onnx_ext.size(), onnx_ext.size(),
                            onnx_ext)
                   == 0)
          {
            long int wt = fileops::file_last_modif(file);
            if (wt > model_t)
              {
                modelf = file;
                model_t = wt;
              }
          }
        else if (file.find(corresp_name) != std::string::npos)
          {
            correspf = file;
          }
      }

    if (modelf.empty())
      {
        logger->error(
            "cannot find .onnx model in repository, make sure there is an "
            ".onnx file in repository {}",
            _repo);
        return 1;
      }

    if (_model.empty())
      _model = modelf;

    if (_corresp.empty())
      _corresp = correspf;

    logger->info("ONNX model resolved: {}", _model);
    if (!_corresp.empty())
      logger->info("Correspondence file resolved: {}", _corresp);

    return 0;
  }
}