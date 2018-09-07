/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#include "caffe2/operators/multi_level_roi_op.h"

namespace caffe2 {

  template<class Context>
  bool MultiLevelRoiOp<Context>::RunOnDevice() {
    size_t nb_items = Input(0).dims()[0];
    const float *rois = Input(0).template data<float>();

    // Determine which FPN level each RoI should map
    std::vector<int> levels(nb_items);
    const float *roi = rois;
    for (int &level : levels) {
      int w = roi[3] - roi[1] + 1;
      int h = roi[4] - roi[2] + 1;
      CAFFE_ENFORCE(w >= 0 && h >= 0);
      level = floor(_canon_level + log2(sqrt(w * h) / _canon_scale + 1e-6));
      level = std::max(_min_level, std::min(level, _max_level));
      roi += 5;
    }

    // Add RoI blobs for multiple FPN levels
    std::vector<size_t> roi_indices(nb_items);
    Output(0)->Resize(std::vector<TIndex>({nb_items}));
    int *idx_restore = Output(0)->template mutable_data<int>();
    int backuped = 0;
    for (int fpn_idx = 1, level = _min_level; level <= _max_level; ++level, ++fpn_idx) {

      // Filter the RoIs
      roi_indices.clear();
      for (size_t roi_idx = 0; roi_idx < nb_items; ++roi_idx) {
	if (levels[roi_idx] == level) {
	  roi_indices.push_back(roi_idx);
	  idx_restore[roi_idx] = backuped++;
	}
      }

      // Store the RoIs
      Output(fpn_idx)->Resize(std::vector<TIndex>({roi_indices.size(), 5}));
      float *fpn = Output(fpn_idx)->template mutable_data<float>();
      for (size_t roi_idx : roi_indices) {
	memcpy(fpn, rois + 5 * roi_idx, 5 * sizeof(float));
	fpn += 5;
      }
    }
    return true;
  }

  REGISTER_CPU_OPERATOR(MultiLevelRoi, MultiLevelRoiOp<CPUContext>);

  OPERATOR_SCHEMA(MultiLevelRoi)
  .NumInputs(1)
  .NumOutputs(2, INT_MAX)
  .Arg("min_level", "(int, 2 by default)"
       " Finest level of the FPN pyramid")
  .Arg("canon_scale", "(int, 224 by default)"
       " Canonical scale for the RoI-to-FPN level mapping")
  .Arg("canon_level", "(int, 4 by default)"
       " Canonical level for the RoI-to-FPN level mapping")
  .Input(0, "roi", "ROIs, size (n, 5), format (image_index, x1, y1, x2, y2)")
  .Output(0, "idx_restore",
	  "Permutation on the concatenation of all rois_fpni, i=min…max, "
	  "such that when applied the RPN RoIs are restored to their original order")
  .Output(1, "fpn", "RPN proposals for a given level, format (image_index, x1, y1, x2, y2)")
  ;
}
