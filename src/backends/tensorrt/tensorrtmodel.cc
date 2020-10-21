// tensorrtmodel.cc ---

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

#include "tensorrtmodel.h"

namespace dd
{
  int TensorRTModel::read_from_repository(
      const std::shared_ptr<spdlog::logger> &logger)
  {
    static std::string deploy = "deploy.prototxt";
    static std::string weights = ".caffemodel";
    static std::string corresp = "corresp";
    static std::string meanf = "mean.binaryproto";

    static std::string model_name = "net_tensorRT";
    static std::string caffe_model_name = model_name + ".proto";
    static std::string onnx_model_name = model_name + ".onnx";

    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(_repo, true, false, false, lfiles);
    if (e != 0)
      {
        logger->error("error reading or listing models in repository {}",
                      _repo);
        return 1;
      }
    std::string deployf, weightsf, correspf, modelf;
    long int weight_t = -1, model_t = -1;
    auto hit = lfiles.begin();
    while (hit != lfiles.end())
      {
        if ((*hit).find(meanf) != std::string::npos)
          {
            _has_mean_file = true;
          }
        else if ((*hit).find(weights) != std::string::npos)
          {
            // stat file to pick the latest one
            long int wt = fileops::file_last_modif((*hit));
            if (wt > weight_t)
              {
                weightsf = (*hit);
                weight_t = wt;
              }
          }
        else if ((*hit).find(corresp) != std::string::npos)
          correspf = (*hit);
        else if ((*hit).find(caffe_model_name) != std::string::npos
                 || (*hit).find(onnx_model_name) != std::string::npos)
          {
            long int wt = fileops::file_last_modif(*hit);
            if (wt > model_t)
              {
                modelf = (*hit);
                model_t = wt;
              }
          }
        else if ((*hit).find("~") != std::string::npos
                 || (*hit).find(".prototxt") == std::string::npos)
          {
            ++hit;
            continue;
          }
        else if ((*hit).find(deploy) != std::string::npos)
          deployf = (*hit);
        ++hit;
      }

    if (_def.empty())
      _def = deployf;
    if (_weights.empty())
      _weights = weightsf;
    if (_corresp.empty())
      _corresp = correspf;
    if (_model.empty())
      _model = modelf;

    return 0;
  }
}
