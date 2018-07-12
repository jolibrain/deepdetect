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

#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    const std::map<std::string, bool> non_trainable_ops({

	{"ConstantFill", true},
	{"GivenTensorFill", true},
	{"XavierFill", true},
	{"GaussianFill", true},
	{"MSRAFill", true},
	{"RangeFill", true},
	{"LengthsRangeFill", true},
	{"DiagonalFill", true},
	{"UniformFill", true},
	{"UniformIntFill", true},
	{"UniqueUniformFill", true},

	{"Accuracy", false},
	{"Cast", false},
	{"Cout", false},
	{"Iter", false},
	{"TensorProtosDBInput", false},
	{"ImageInput", false},
	{"TimePlot", false},
	{"ShowWorst", false},
	{"NHWC2NCHW", false}

      });

    const std::map<std::string, std::set<int> > trainable_ops({
	{"Add", {}},
	{"AffineScale", {}},
	{"AveragedLoss", {}},
	{"AveragePool", {}},
	{"BackMean", {}},
	{"Concat", {}},
	{"Conv", {}},
	{"Diagonal", {}},
	{"Dropout", {}},
	{"EnsureCPUOutput", {}},
	{"FC", {}},
	{"LabelCrossEntropy", {}},
	{"LRN", {}},
	{"MaxPool", {}},
	{"Mul", {}},
	{"RecurrentNetwork", {}},
	{"Relu", {}},
	{"Reshape", {}},
	{"Scale", {}},
	{"Slice", {}},
	{"Softmax", {}},
	{"SpatialBN", {3, 4}},
	{"SquaredL2", {}},
	{"SquaredL2Channel", {}},
	{"StopGradient", {}},
	{"Sub", {}},
	{"Sum", {}}
      });

    const std::string mean_square_suffix("_meansq");
    const std::string momentum_suffix("_moment");
    const std::string gradient_suffix("_grad");
    const std::string scale_suffix("_scale");

    const std::string blob_lr("lr");
    const std::string blob_one("one");
    const std::string blob_iter("iter");
    const std::string blob_xent("xent");
    const std::string blob_loss("loss");
    const std::string blob_loss_scale(blob_loss + scale_suffix);

  }
}
