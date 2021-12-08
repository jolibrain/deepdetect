// Copyright (C) 2021 Jolibrain http://www.jolibrain.com

// Author: Louis Jean <louis.jean@jolibrain.com>

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

#include <vector>
#include <algorithm>

namespace dd
{
  namespace yolo_utils
  {
    /** Convert from format:
     * unsorted bbox*4 | objectness | class softmax*n_classes
     * to format:
     * sorted batch id | class_id | class confidence | bbox * 4*/
    static std::vector<float>
    parse_yolo_output(const std::vector<float> &model_out, size_t batch_size,
                      size_t top_k, size_t n_classes, size_t im_width,
                      size_t im_height)
    {
      std::vector<float> vals;
      vals.reserve(batch_size * top_k * 7);
      size_t step = n_classes + 5;
      auto batch_it = model_out.begin();

      for (size_t batch = 0; batch < batch_size; ++batch)
        {
          std::vector<std::vector<float>> result;
          result.reserve(top_k);
          auto end_it = batch_it + top_k * step;

          for (; batch_it != end_it; batch_it += step)
            {
              // get class id & confidence
              auto max_batch_it
                  = std::max_element(batch_it + 5, batch_it + step);
              float cls_pred = std::distance(batch_it + 5, max_batch_it);
              float prob = *max_batch_it * (*(batch_it + 4));

              // convert center, dims to xyxy
              float xc = *batch_it, yc = *(batch_it + 1), w = *(batch_it + 2),
                    h = *(batch_it + 3);
              result.push_back(std::vector<float>{
                  0, cls_pred, prob, (xc - w / 2) / (im_width - 1),
                  (yc - h / 2) / (im_height - 1),
                  (xc + w / 2) / (im_width - 1),
                  (yc + h / 2) / (im_height - 1) });
            }

          std::sort(result.begin(), result.end(),
                    [](const std::vector<float> &a,
                       const std::vector<float> &b) { return a[2] > b[2]; });

          for (auto &val : result)
            {
              vals.insert(vals.end(), val.begin(), val.end());
            }
          batch_it = end_it;
        }
      return vals;
    }
  }
}
