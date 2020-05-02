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

#ifndef STATISTICS_H
#define STATISTICS_H

#include <chrono>
#include <mutex>

#include "apidata.h"

namespace dd
{
  class ServiceStats
  {

  public:
    ServiceStats()
    {
    }

    ServiceStats(ServiceStats &stats)
    {
      // NOTE(sileht) : Do we really want to have all stats copied ?
      _inference_count = stats._inference_count;

      _predict_success = stats._predict_success;
      _predict_failure = stats._predict_failure;
      _predict_tstart = stats._predict_tstart;

      _transform_tstart = stats._transform_tstart;

      _avg_batch_size = stats._avg_batch_size;
      _avg_predict_duration = stats._avg_predict_duration;
      _avg_transform_duration = stats._avg_transform_duration;
    }

    ~ServiceStats()
    {
    }

    void inc_inference_count(const int &l);

    void transform_start();
    void transform_end();

    void predict_start();
    void predict_end(bool succeed);

    void to(APIData &ad) const;

  private:
    int _inference_count = 0;

    int _predict_success = 0;
    int _predict_failure = 0;

    std::chrono::steady_clock::time_point _predict_tstart;
    std::chrono::duration<double> _predict_total_duration;

    std::chrono::steady_clock::time_point _transform_tstart;
    std::chrono::duration<double> _transform_total_duration;

    double _avg_batch_size = -1;
    double _avg_predict_duration = -1;
    double _avg_transform_duration = -1;

    mutable std::mutex _mutex; /**< mutex for converting to APIData. */
  };
};

#endif
