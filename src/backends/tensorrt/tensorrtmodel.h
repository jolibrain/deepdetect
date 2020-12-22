/* tensorrtmodel.h ---  */

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

#ifndef TENSORRTMODEL_H
#define TENSORRTMODEL_H

#include "dd_spdlog.h"
#include "mlmodel.h"
#include "apidata.h"

namespace dd
{
  class TensorRTModel : public MLModel
  {
  public:
    TensorRTModel() : MLModel()
    {
    }
    TensorRTModel(const APIData &ad, APIData &adg,
                  const std::shared_ptr<spdlog::logger> &logger)
        : MLModel(ad, adg, logger)
    {
      if (ad.has("repository"))
        {
          this->_repo = ad.get("repository").get<std::string>();
          read_from_repository(spdlog::get("api"));
          read_corresp_file();
        }
    }
    TensorRTModel(const std::string &repo) : MLModel(repo)
    {
    }

    ~TensorRTModel()
    {
    }

    int read_from_repository(const std::shared_ptr<spdlog::logger> &logger);

    inline bool is_caffe_source() const
    {
      return _source_type == "caffe";
    }

    inline bool is_onnx_source() const
    {
      return _source_type == "onnx";
    }

    std::string _model;
    std::string _def;
    std::string _weights;
    bool _has_mean_file = false;
    std::string _source_type = "caffe"; // or "onnx"
  };
}

#endif
