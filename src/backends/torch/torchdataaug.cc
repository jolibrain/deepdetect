/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
 * Authors: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#include "torchdataaug.h"
#include "torchdataset.h"

namespace dd
{
  namespace
  {
    void
    tensor_targets_to_bbox_vectors(const std::vector<torch::Tensor> &targets,
                                   std::vector<std::vector<float>> &bboxes,
                                   std::vector<int> &classes)
    {
      torch::Tensor t = targets[0];
      torch::Tensor c = targets[1];
      int nbbox = t.size(0);
      bboxes.clear();
      classes.clear();
      bboxes.reserve(nbbox);
      classes.reserve(nbbox);
      for (int bb = 0; bb < nbbox; ++bb)
        {
          std::vector<float> bbox;
          bbox.reserve(4);
          for (int d = 0; d < 4; ++d)
            bbox.push_back(t[bb][d].item<float>());
          bboxes.push_back(bbox);
          classes.push_back(c[bb].item<int>());
        }
    }

    void bbox_vectors_to_tensor_targets(
        const std::vector<std::vector<float>> &bboxes,
        const std::vector<int> &classes, std::vector<torch::Tensor> &targets)
    {
      std::vector<torch::Tensor> tbboxes;
      std::vector<torch::Tensor> tclasses;
      tbboxes.reserve(bboxes.size());
      tclasses.reserve(classes.size());
      TorchDataset td;
      for (size_t bb = 0; bb < bboxes.size(); ++bb)
        {
          std::vector<double> fbbox(bboxes.at(bb).begin(),
                                    bboxes.at(bb).end());
          tbboxes.push_back(td.target_to_tensor(fbbox));
          tclasses.push_back(td.target_to_tensor(classes.at(bb)));
        }
      targets = { torch::stack(tbboxes), torch::cat(tclasses) };
    }
  }

  void
  TorchImgRandAugCV::augment_with_bbox(cv::Mat &src,
                                       std::vector<torch::Tensor> &targets)
  {
    std::vector<std::vector<float>> bboxes;
    std::vector<int> classes;
    tensor_targets_to_bbox_vectors(targets, bboxes, classes);
    ImgRandAugCV::augment_with_bbox(src, bboxes, classes);
    bbox_vectors_to_tensor_targets(bboxes, classes, targets);
  }

  void TorchImgRandAugCV::augment_test_with_bbox(
      cv::Mat &src, std::vector<torch::Tensor> &targets)
  {
    std::vector<std::vector<float>> bboxes;
    std::vector<int> classes;
    tensor_targets_to_bbox_vectors(targets, bboxes, classes);
    ImgRandAugCV::augment_test_with_bbox(src, bboxes, classes);
    bbox_vectors_to_tensor_targets(bboxes, classes, targets);
  }
}
