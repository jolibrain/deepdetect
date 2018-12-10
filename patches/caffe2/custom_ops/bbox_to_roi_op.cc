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

#include "caffe2/operators/bbox_to_roi_op.h"

namespace caffe2 {

  template<class Context>
  bool BBoxToRoiOp<Context>::RunOnDevice() {
    size_t nb_items = Input(0).dim(0);
    Output(0)->Resize(std::vector<long int>({nb_items, 5}));

    const float *bbox = Input(0).template data<float>();
    const float *infos = Input(1).template data<float>();
    float *roi = Output(0)->template mutable_data<float>();

    int batch_size = 1;
    vector<float> batch_splits_default(1, nb_items);
    const float *batch_splits_data = batch_splits_default.data();
    if (InputSize() > 2) {
      batch_size = Input(2).dim(0);
      batch_splits_data = Input(2).template data<float>();
    }
    std::vector<float> batch_splits(batch_splits_data, batch_splits_data + batch_size);

    for (int batch = 0; batch < batch_size; ++batch) {
      float scale = infos[2];
      infos += 3;
      for (int item = batch_splits[batch]; item--;) {
	*roi++ = batch;
	for (int i = 0; i < 4; ++i) {
	  *roi++ = *bbox++ * scale;
	}
      }
    }

    return true;
  }

  REGISTER_CPU_OPERATOR(BBoxToRoi, BBoxToRoiOp<CPUContext>);

  OPERATOR_SCHEMA(BBoxToRoi)
  .NumInputs(2, 3)
  .NumOutputs(1)
  .Input(0, "bbox_nms", "Filtered boxes, size (n, 4), format (x1, y1, x2, y2)")
  .Input(1, "im_info", "Image info, size (img_count, 3), format (height, width, scale)")
  .Input(2, "batch_splits",
	 "Tensor of shape (batch_size) with each element denoting "
	 "the number of boxes belonging to the corresponding image in batch. "
	 "Sum should add up to total count of boxes.")
  .Output(0, "roi", "Scaled ROIs, size (n, 5), format (image_index, x1, y1, x2, y2)")
  ;
}
