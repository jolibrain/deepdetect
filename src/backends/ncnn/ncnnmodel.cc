/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#include "ncnnmodel.h"
#include <filesystem>
#include <set>

namespace dd
{
  void NCNNModel::read_from_repository()
  {
    namespace fs = std::filesystem;
    std::set<fs::path> params, weights;
    for (const auto &entry : fs::directory_iterator(_repo))
      {
        if (!entry.is_regular_file())
          continue;
        const auto path = entry.path();
        const auto stem = path.parent_path() / path.stem();
        if (stem.extension() == ".pnnx")
          continue;
        if (path.extension() == ".param")
          params.insert(stem);
        else if (path.extension() == ".bin")
          weights.insert(stem);
        else if (path.filename() == "corresp.txt"
                 || path.filename() == "corresp")
          {
            if (!_corresp.empty())
              throw MLLibBadParamException("multiple NCNN label files");
            _corresp = path.string();
          }
      }
    if (params.size() != 1 || params != weights)
      throw MLLibBadParamException(
          "expected one matching NCNN .param/.bin pair. Export Caffe or "
          "patched "
          "models to standard NCNN before creating a service.");
    _params = params.begin()->string() + ".param";
    _weights = weights.begin()->string() + ".bin";
  }
}
