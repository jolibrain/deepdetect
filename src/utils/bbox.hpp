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
    double area(const std::vector<double> &bbox)
    {
      return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    }

    std::vector<double> intersect(const std::vector<double> &bbox1,
                                  const std::vector<double> &bbox2)
    {
      std::vector<double> inter{
        std::max(bbox1[0], bbox2[0]),
        std::max(bbox1[1], bbox2[1]),
        std::min(bbox1[2], bbox2[2]),
        std::min(bbox1[3], bbox2[3]),
      };
      // if xmin > xmax or ymin > ymax, intersection is empty
      if (inter[0] >= inter[2] || inter[1] >= inter[3])
        {
          return { 0., 0., 0., 0. };
        }
      else
        return inter;
    }

    double iou(const std::vector<double> &bbox1,
               const std::vector<double> &bbox2)
    {
      double a1 = area(bbox1);
      double a2 = area(bbox2);
      auto inter = intersect(bbox1, bbox2);
      double ainter = area(inter);
      return ainter / (a1 + a2 - ainter);
    }
  }
}

#endif // DD_UTILS_BBOX_HPP
