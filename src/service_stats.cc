
/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author: Mehdi Abaakouk <mehdi.abaakouk@jolibrain.com>
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

#include <chrono>

#include "apidata.h"
#include "service_stats.h"

namespace dd
{

  void ServiceStats::inc_inference_count(const int &l)
  {
    _inference_count += l;
  }
  void ServiceStats::transform_start()
  {
    _transform_tstart = std::chrono::steady_clock::now();
  }
  void ServiceStats::transform_end()
  {
    auto tend = std::chrono::steady_clock::now();
    _transform_total_duration += tend - _transform_tstart;
  }

  void ServiceStats::predict_start()
  {
    _predict_tstart = std::chrono::steady_clock::now();
  }

  void ServiceStats::predict_end(bool succeed)
  {
    std::lock_guard<std::mutex> lock(_mutex);

    if (succeed)
      _predict_success++;
    else
      _predict_failure++;

    auto tend = std::chrono::steady_clock::now();
    _predict_total_duration += tend - _predict_tstart;

    int _predict_count = _predict_success + _predict_failure;
    _avg_batch_size = _inference_count / static_cast<double>(_predict_count);
    _avg_transform_duration = _transform_total_duration.count()
                              / static_cast<double>(_predict_count);
    _avg_predict_duration = _predict_total_duration.count()
                            / static_cast<double>(_predict_count);
  }

  void ServiceStats::to(APIData &ad) const
  {
    std::lock_guard<std::mutex> lock(_mutex);

    APIData stats;

    stats.add("inference_count", _inference_count);
    stats.add("predict_success", _predict_success);
    stats.add("predict_failure", _predict_failure);
    stats.add("predict_count", _predict_success + _predict_failure);
    stats.add("avg_batch_size", _avg_batch_size);
    stats.add("avg_predict_duration", _avg_predict_duration);
    stats.add("avg_transform_duration", _avg_transform_duration);

    ad.add("service_stats", stats);
  }

}
