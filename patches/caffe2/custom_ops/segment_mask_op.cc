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

#include <opencv2/imgproc/imgproc.hpp>
#include "caffe2/operators/segment_mask_op.h"

namespace {
  enum Index {
    score,
    bbox,
    cls,
    mask,
    im_info,
    batch_splits,
    TOTAL
  };
}

namespace caffe2 {

  template<class Context>
  bool SegmentMaskOp<Context>::RunOnDevice() {
    const float *bbox = Input(Index::bbox).template data<float>();
    const float *cls = Input(Index::cls).template data<float>();
    const float *mask = Input(Index::mask).template data<float>();
    const float *infos = Input(Index::im_info).template data<float>();

    // Input shape
    const vector<TIndex> &dims = Input(Index::mask).dims();
    int nb_items = dims[0];
    int layers = dims[1];
    int mask_size = dims[2];
    int mask_bytes = mask_size * sizeof(float);
    CAFFE_ENFORCE(dims[3] == mask_size);
    int layer_size = mask_size * mask_size;
    int layers_size = layers * layer_size;
    int batch_size = Input(Index::im_info).dim(0);

    // Output shape
    int img_h = infos[0] / infos[2];
    int img_w = infos[1] / infos[2];
    const float *infos_ptr = infos;
    for (int i = 1; i < batch_size; ++i) {
      infos_ptr += 3;
      CAFFE_ENFORCE(static_cast<int>(infos_ptr[0] / infos_ptr[2]) == img_h);
      CAFFE_ENFORCE(static_cast<int>(infos_ptr[1] / infos_ptr[2]) == img_w);
    }
    int img_size = img_h * img_w;
    Output(Index::mask)->Resize(std::vector<TIndex>({nb_items, img_h, img_w}));
    float *segm_mask = Output(Index::mask)->template mutable_data<float>();
    std::memset(segm_mask, 0, nb_items * img_size);

    // To work around an issue with cv2.resize
    // (see the 'segm_results' function in Detectron/detectron/core/test.py)
    int padded_mask_size = mask_size + 2;
    cv::Mat padded_mask(padded_mask_size, padded_mask_size, CV_32F);
    padded_mask.setTo(0);
    float scale = static_cast<float>(padded_mask_size) / mask_size;

    // Find the mask corresponding to each item
    for (int item = 0; item < nb_items; ++item) {
      int layer = layers > 1 ? cls[item] : 0;
      const float *mask_data = mask + layer * layer_size;

      // Workaround
      int xmin = bbox[0], ymin = bbox[1], xmax = bbox[2], ymax = bbox[3];
      float w_half = (xmax - xmin) * 0.5 * scale;
      float h_half = (ymax - ymin) * 0.5 * scale;
      float x_center = (xmax + xmin) * 0.5;
      float y_center = (ymax + ymin) * 0.5;
      xmin = x_center - w_half;
      ymin = y_center - h_half;
      xmax = x_center + w_half;
      ymax = y_center + h_half;
      int w = std::max(xmax - xmin + 1, 1);
      int h = std::max(ymax - ymin + 1, 1);
      int segm_xmin = std::max(xmin, 0);
      int segm_ymin = std::max(ymin, 0);
      int segm_w = std::min(xmax + 1, img_w) - segm_xmin;
      int segm_h = std::min(ymax + 1, img_h) - segm_ymin;
      int mask_xmin = segm_xmin - xmin;
      int mask_ymin = segm_ymin - ymin;
      // Start to fill the padded mask at the coordinate (1,1) instead of (0,0)
      float *padded_mask_data =
	reinterpret_cast<float*>(padded_mask.data) + padded_mask_size + 1;
      for (int y = 0; y < mask_size; ++y) {
	std::memcpy(padded_mask_data, mask_data, mask_bytes);
	padded_mask_data += padded_mask_size;
	mask_data += mask_size;
      }

      // Resize
      cv::Mat resized_mask;
      cv::resize(padded_mask, resized_mask, cv::Size(w, h));
      mask_data = reinterpret_cast<float*>(resized_mask.data);

      // Start from the right offsets
      float *segm_mask_data = segm_mask + segm_ymin * img_w + segm_xmin;
      mask_data += mask_ymin * w + mask_xmin;
      for (int y = 0; y < segm_h; ++y) {
	for (int x = 0; x < segm_w; ++x) {
	  // Apply threshold
	  segm_mask_data[x] = mask_data[x] > _thresh_bin;
	}
	segm_mask_data += img_w;
	mask_data += w;
      }

      // Switch to the next item
      segm_mask += img_size;
      mask += layers_size;
      bbox += 4;
    }

    // Forward the blobs
    const std::set<int> indices({
	Index::score, Index::bbox, Index::cls, Index::im_info, Index::batch_splits
    });
    for (int i : indices) {
      const auto &input = Input(i);
      auto &output = *Output(i);
      output.ResizeLike(input);
      output.ShareData(input);
    }

    return true;
  }

  REGISTER_CPU_OPERATOR(SegmentMask, SegmentMaskOp<CPUContext>);

  OPERATOR_SCHEMA(SegmentMask)
  .NumInputs(Index::TOTAL)
  .NumOutputs(Index::TOTAL)
  .AllowInplace({
      {Index::score, Index::score},
      {Index::bbox, Index::bbox},
      {Index::cls, Index::cls},
      {Index::im_info, Index::im_info},
      {Index::batch_splits, Index::batch_splits}
    })
  .Arg("thresh_bin", "(float, 0.5 by default)"
       " Binarization threshold for converting soft masks to hard masks")
  .Input(Index::score, "score_nms", "Filtered scores, size (n)")
  .Input(Index::bbox, "bbox_nms", "Filtered boxes, size (n, 4)")
  .Input(Index::cls, "class_nms", "Class id for each filtered score/box, size (n)")
  .Input(Index::mask, "mask_pred", "Compressed masks, size (n, nbclasses, length, length)")
  .Input(Index::im_info, "im_info",
	 "Image info, size (img_count, 3), format (height, width, scale)")
  .Input(Index::batch_splits, "batch_splits",
	 "Tensor of shape (batch_size) with each element denoting "
	 "the number of boxes belonging to the corresponding image in batch. "
	 "Sum should add up to total count of boxes.")
  .Output(Index::score, "score_nms", "Input propagation")
  .Output(Index::bbox, "bbox_nms", "Input propagation")
  .Output(Index::cls, "class_nms", "Input propagation")
  .Output(Index::mask, "mask", "Scaled masks, size (n, h, w")
  .Output(Index::im_info, "im_info", "Input propagation")
  .Output(Index::batch_splits, "batch_splits", "Input propagation")
  ;
}
