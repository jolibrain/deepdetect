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

    // Parametrable functions

    template <class OpDef>
    GetOpInputFct _fill_trainable_inputs() {
      return [](const caffe2::OperatorDef &op, std::set<std::string> &inputs) {
	auto index = OpDef::computed.begin();
	auto end = OpDef::computed.end();
	int i = 0;
	for (const std::string &input : op.input()) {
	  if (index != end && *index == i++) {
	    ++index;
	  } else {
	    inputs.insert(input);
	  }
	}
      };
    }

    template <class OpDef>
    GetOpInputFct _fill_computed_inputs() {
      return [](const caffe2::OperatorDef &op, std::set<std::string> &inputs) {
	auto index = OpDef::computed.begin();
	auto end = OpDef::computed.end();
	int i = 0;
	for (const std::string &input : op.input()) {
	  if (index != end && *index == i++) {
	    ++index;
	    inputs.insert(input);
	  }
	}
      };
    }

    template <class OpDef>
    GetOpOutputFct _fill_needed_outputs() {
      return [](const caffe2::OperatorDef &op, std::vector<std::string> &outputs) {

	// Skip the already existing outputs
	const auto &op_inputs = op.input();
	const auto &op_outputs = op.output();
	int nb_outputs = op_outputs.size();
	for (const auto &output : OpDef::needed) {
	  if (nb_outputs-- > 0) {
	    continue;
	  }

	  // Should not be a prediction output
	  CAFFE_ENFORCE(output.train_only);

	  // Get the base name
	  std::string base_name;
	  if (output.use_input) {
	    base_name = op_inputs[output.blob_index];
	  } else {
	    base_name = op_outputs[output.blob_index];
	  }

	  // Store the new blob
	  outputs.push_back(base_name + output.blob_suffix);
	}
      };
    }

    template <class OpDef>
    inline GetOpBlobsFcts _operator_blob_getters() {
      return std::make_tuple
	(_fill_trainable_inputs<OpDef>(),
	 _fill_computed_inputs<OpDef>(),
	 _fill_needed_outputs<OpDef>());
    }

    // Classes used as parameters

    class OutputDef {
    public:
      bool train_only;
      bool use_input;
      int blob_index;
      std::string blob_suffix;
      OutputDef(bool train, bool input, int index, const std::string &suffix):
	train_only(train), use_input(input), blob_index(index), blob_suffix(suffix) {}
    };

    // Default
    class DefaultOperatorDef { public:
      static const std::set<int> computed;
      static const std::vector<OutputDef> needed;
    };
    const std::set<int> DefaultOperatorDef::computed;
    const std::vector<OutputDef> DefaultOperatorDef::needed;

    // Incstance normalization
    class InstanceNormDef { public:
      static const std::set<int> computed;
      static const std::vector<OutputDef> needed;
    };
    const std::set<int> InstanceNormDef::computed;
    const std::vector<OutputDef> InstanceNormDef::needed({
        OutputDef(false, false, 0, "" ), // output (default)
	OutputDef(true, false, 0, "_sm"), // saved_min
	OutputDef(true, false, 0, "_siv") // saved_inv_stdev
    });

    // Batch normalization
    class SpatialBNDef { public:
      static const std::set<int> computed;
      static const std::vector<OutputDef> needed;
    };
    const std::set<int> SpatialBNDef::computed({3, 4}); // Mean & Var
    const std::vector<OutputDef> SpatialBNDef::needed({
        OutputDef(false, false, 0, ""), // output (default)
	OutputDef(true, true, 3, ""), // mean (inplace)
	OutputDef(true, true, 4, ""), // var (inplace)
	OutputDef(true, false, 0, "_sm"), // saved_mean
	OutputDef(true, false, 0, "_siv") // saved_var
    });

    // Register

    const GetOpBlobsFcts _get_blobs = _operator_blob_getters<DefaultOperatorDef>();
    const GetOpBlobsFcts _get_blobs_instance_norm = _operator_blob_getters<InstanceNormDef>();
    const GetOpBlobsFcts _get_blobs_spatial_bn = _operator_blob_getters<SpatialBNDef>();

    const std::map<std::string, GetOpBlobsFcts> trainable_ops({
	{"Add", _get_blobs},
	{"AffineScale", _get_blobs},
	{"AveragedLoss", _get_blobs},
	{"AveragePool", _get_blobs},
	{"BackMean", _get_blobs},
	{"Concat", _get_blobs},
	{"Conv", _get_blobs},
	{"Diagonal", _get_blobs},
	{"Dropout", _get_blobs},
	{"EnsureCPUOutput", _get_blobs},
	{"FC", _get_blobs},
	{"InstanceNorm", _get_blobs_instance_norm},
	{"LabelCrossEntropy", _get_blobs},
	{"LRN", _get_blobs},
	{"MaxPool", _get_blobs},
	{"Mul", _get_blobs},
	{"RecurrentNetwork", _get_blobs},
	{"Relu", _get_blobs},
	{"Reshape", _get_blobs},
	{"Scale", _get_blobs},
	{"Slice", _get_blobs},
	{"Softmax", _get_blobs},
	{"SpatialBN", _get_blobs_spatial_bn},
	{"SquaredL2", _get_blobs},
	{"SquaredL2Channel", _get_blobs},
	{"StopGradient", _get_blobs},
	{"Sub", _get_blobs},
	{"Sum", _get_blobs}
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

  }
}
