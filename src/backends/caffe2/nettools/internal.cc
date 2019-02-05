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

    // Note if another operator needs to be registered here :
    // A quick way to check if it has or not a gradient is to check if its type is in
    // caffe2::GradientRegistry()->Keys()

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

    void TrainableOp::get_trainable_inputs(const caffe2::OperatorDef &op,
					   std::set<std::string> &inputs) const {
      auto index = _computed_inputs.begin();
      auto end = _computed_inputs.end();
      int i = 0;
      for (const std::string &input : op.input()) {
	if (index != end && *index == i++) {
	  ++index;
	} else {
	  inputs.insert(input);
	}
      }
    }

    void TrainableOp::get_computed_inputs(const caffe2::OperatorDef &op,
					  std::set<std::string> &inputs) const {
      auto index = _computed_inputs.begin();
      auto end = _computed_inputs.end();
      int i = 0;
      for (const std::string &input : op.input()) {
	if (index != end && *index == i++) {
	  ++index;
	  inputs.insert(input);
	}
      }
    }

    void TrainableOp::get_needed_outputs(const caffe2::OperatorDef &op,
					 std::vector<std::string> &outputs) const {
      // Skip the already existing outputs
      const auto &op_inputs = op.input();
      const auto &op_outputs = op.output();
      int nb_outputs = op_outputs.size();
      for (const auto &output : _outputs) {
	if (nb_outputs-- > 0) {
	  continue;
	}

	// Should not be a prediction output
	CAFFE_ENFORCE(output._training_only);

	// Get the base name
	std::string base_name;
	if (output._based_on_input) {
	  base_name = op_inputs[output._based_on_index];
	} else {
	  base_name = op_outputs[output._based_on_index];
	}

	// Store the new blob
	outputs.push_back(base_name + output._suffix);
      }
    }

    void TrainableOp::get_trainable_overwrites(const caffe2::OperatorDef &op,
					       std::set<int> &outputs) const {
      int out_size = std::min(static_cast<int>(_outputs.size()), op.output().size());
      int in_size = op.input().size();
      for (int out_idx = 0; out_idx < out_size; ++out_idx) {
	int can_overwrite = _outputs[out_idx]._can_overwrite;
	if (can_overwrite >= 0 && can_overwrite < in_size &&
	    op.output(out_idx) == op.input(can_overwrite)) {
	  outputs.insert(out_idx);
	}
      }
    }

    // Inplace
    class TrainableInplaceOp : public TrainableOp { public:
      TrainableInplaceOp(): TrainableOp({
	}, {
	  Output(false, false, 0, "", 0) // No partcular restriction, but can be inplace
	}) {}
    };

    // Instance normalization
    class TrainableInstanceNormOp : public TrainableOp { public:
      TrainableInstanceNormOp(): TrainableOp({
	}, {
	  Output(false, false, 0, "", -1), // output (default)
	  Output(true, false, 0, "_sm", -1), // saved_min
	  Output(true, false, 0, "_siv", -1) // saved_inv_stdev
	}) {}
    };

    // Batch normalization
    class TrainableSpatialBNOp : public TrainableOp { public:
      TrainableSpatialBNOp(): TrainableOp({
	  3, 4 // Mean & Var
	}, {
	  Output(false, false, 0, "", -1), // output (default)
	  Output(true, true, 3, "", 3), // mean (inplace)
	  Output(true, true, 4, "", 4), // var (inplace)
	  Output(true, false, 0, "_sm", -1), // saved_mean
	  Output(true, false, 0, "_siv", -1) // saved_var
	}) {}
    };

    // Register

    const TrainableOp _default_op;
    const TrainableOp _inplace_op = TrainableInplaceOp();
    const TrainableOp _instance_norm_op = TrainableInstanceNormOp();
    const TrainableOp _spatial_bn_op = TrainableSpatialBNOp();

    const std::map<std::string, TrainableOp> trainable_ops({
	{"Add", _default_op},
	{"AffineScale", _default_op},
	{"AveragedLoss", _default_op},
	{"AveragePool", _default_op},
	{"BackMean", _default_op},
	{"Concat", _default_op},
	{"Conv", _default_op},
	{"Diagonal", _default_op},
	{"Dropout", _inplace_op},
	{"EnsureCPUOutput", _default_op},
	{"FC", _default_op},
	{"InstanceNorm", _instance_norm_op},
	{"LabelCrossEntropy", _default_op},
	{"LRN", _default_op},
	{"MaxPool", _default_op},
	{"Mul", _default_op},
	{"RecurrentNetwork", _default_op},
	{"Relu", _inplace_op},
	{"Reshape", _default_op},
	{"Scale", _default_op},
	{"Sigmoid", _default_op},
	{"Slice", _default_op},
	{"Softmax", _default_op},
	{"SpatialBN", _spatial_bn_op},
	{"SquaredL2", _default_op},
	{"SquaredL2Channel", _default_op},
	{"StopGradient", _default_op},
	{"Sub", _default_op},
	{"Sum", _inplace_op}
      });

    const std::string batch_splits_suffix("_batch_splits");
    const std::string force_device_suffix("_force_device");
    const std::string mean_square_suffix("_meansq");
    const std::string momentum_suffix("_momentum");
    const std::string gradient_suffix("_grad");
    const std::string scale_suffix("_scale");

    const std::string blob_lr("lr");
    const std::string blob_one("one");
    const std::string blob_iter("iter");
    const std::string blob_xent("xent");
    const std::string blob_loss("loss");
    const std::string blob_loss_scale(blob_loss + scale_suffix);

    bool is_trainable(const caffe2::OperatorDef &op, const TrainableOp* &train_op) {
      const auto &it = trainable_ops.find(op.type());
      if (it == trainable_ops.end()) {
	return false;
      }
      train_op = &it->second;
      return true;
    }
  }
}
