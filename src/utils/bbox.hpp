/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
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

#ifndef DD_UTILS_BBOX_HPP
#define DD_UTILS_BBOX_HPP

#include <vector>

namespace dd
{
  namespace bbox_utils
  {
    template <typename T> inline T area(const std::vector<T> &bbox)
    {
      return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    }

    template <typename T>
    inline std::vector<T> intersect(const std::vector<T> &bbox1,
                                    const std::vector<T> &bbox2)
    {
      std::vector<T> inter{
        std::max(bbox1[0], bbox2[0]),
        std::max(bbox1[1], bbox2[1]),
        std::min(bbox1[2], bbox2[2]),
        std::min(bbox1[3], bbox2[3]),
      };
      // if xmin > xmax or ymin > ymax, intersection is empty
      if (inter[0] >= inter[2] || inter[1] >= inter[3])
        {
          return { T(0), T(0), T(0), T(0) };
        }
      else
        return inter;
    }

    template <typename T>
    inline T iou(const std::vector<T> &bbox1, const std::vector<T> &bbox2)
    {
      auto a1 = area(bbox1);
      auto a2 = area(bbox2);
      auto inter = intersect(bbox1, bbox2);
      auto ainter = area(inter);
      return ainter / (a1 + a2 - ainter);
    }

    /** bboxes: list of bboxes in the format { xmin, ymin, xmax, ymax } sorted
     * by decreasing confidence
     *
     * picked: vector used as output containing indices of bboxes kept by nms.
     */
    template <typename T>
    inline void nms_sorted_bboxes(const std::vector<std::vector<T>> &bboxes,
                                  std::vector<size_t> &picked, T nms_threshold)
    {
      picked.clear();
      const size_t n = bboxes.size();

      for (size_t i = 0; i < n; i++)
        {
          const std::vector<T> &bbox_a = bboxes[i];

          bool keep = true;
          for (size_t j = 0; j < picked.size(); j++)
            {
              const std::vector<T> &bbox_b = bboxes[picked[j]];

              // intersection over union
              auto iou = bbox_utils::iou(bbox_a, bbox_b);
              if (iou > nms_threshold)
                keep = false;
            }

          if (keep)
            picked.push_back(i);
        }
    }

    inline void nms_sorted_bboxes(std::vector<APIData> &bboxes,
                                  std::vector<double> &probs,
                                  std::vector<std::string> &cats,
                                  double nms_threshold, int best_bbox)
    {
      std::vector<std::vector<double>> sorted_boxes;
      std::vector<size_t> picked;

      for (size_t l = 0; l < bboxes.size(); ++l)
        {
          std::vector<double> bbox_vec{ bboxes[l].get("xmin").get<double>(),
                                        bboxes[l].get("ymin").get<double>(),
                                        bboxes[l].get("xmax").get<double>(),
                                        bboxes[l].get("ymax").get<double>() };
          sorted_boxes.push_back(bbox_vec);
        }
      // We assume that bboxes are already sorted in model output

      bbox_utils::nms_sorted_bboxes(sorted_boxes, picked, nms_threshold);
      std::vector<APIData> nbboxes;
      std::vector<double> nprobs;
      std::vector<std::string> ncats;

      for (size_t pick : picked)
        {
          nbboxes.push_back(bboxes.at(pick));
          nprobs.push_back(probs.at(pick));
          ncats.push_back(cats.at(pick));

          if (best_bbox > 0
              && nbboxes.size() >= static_cast<size_t>(best_bbox))
            break;
        }

      bboxes = nbboxes;
      probs = nprobs;
      cats = ncats;
    }
  }
}

#endif // DD_UTILS_BBOX_HPP
