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

    /*
     *  Device management
     */

#ifdef CPU_ONLY
    std::string get_device_prefix(const caffe2::DeviceOption &) { return ""; }
#else
    std::string device_id_to_prefix(int id) {
      return "gpu_" + std::to_string(id) + "/";
    }
    std::string get_device_prefix(const caffe2::DeviceOption &option) {
      return option.device_type() == caffe2::CUDA ?
	device_id_to_prefix(option.cuda_gpu_id()) : "";
    }
#endif

#define ADD_EXTERNAL(type)						\
    void add_external_##type(ScopedNet &net, const std::string &type) { \
      for (const caffe2::DeviceOption &option : net._devices) {		\
	net._net.get().add_external_##type(get_device_prefix(option) + type); \
      }									\
    }

    ADD_EXTERNAL(input)
    ADD_EXTERNAL(output)

    void set_net_device(caffe2::NetDef &net, const caffe2::DeviceOption &device) {
      for (caffe2::OperatorDef &op : *net.mutable_op()) {
	op.mutable_device_option()->CopyFrom(device);
      }
    }

    /*
     *  Protobuffer manipulation
     */

    void del_arg(caffe2::OperatorDef &op, const std::string &name) {
      auto &args = *op.mutable_arg();
      for (auto it = args.begin(); it != args.end(); ++it) {
	if (it->name() == name) {
	  args.erase(it);
	  return;
	}
      }
    }

#define REGISTER_ARG_SETTER(type, code)				\
    caffe2::Argument &add_arg(caffe2::OperatorDef &op,		\
			      const std::string& name,		\
			      const type &value) {		\
      caffe2::Argument &arg = *op.add_arg();			\
      arg.set_name(name);					\
      code;							\
      return arg;						\
    }								\
    caffe2::Argument &replace_arg(caffe2::OperatorDef &op,	\
				  const std::string& name,	\
				  const type &value) {		\
      del_arg(op, name);					\
      return add_arg(op, name, value);				\
    }

#define REGISTER_ARG_SETTER_TYPE(type, name) REGISTER_ARG_SETTER(type, arg.set_##name(value))
#define REGISTER_ARG_SETTER_CONTAINER(type, name)			\
    REGISTER_ARG_SETTER(std::vector<type>, for (const type &v : value) arg.add_##name(v))

    REGISTER_ARG_SETTER_TYPE(int, i)
    REGISTER_ARG_SETTER_TYPE(float, f)
    REGISTER_ARG_SETTER_TYPE(std::string, s)
    REGISTER_ARG_SETTER_CONTAINER(int, ints)
    REGISTER_ARG_SETTER_CONTAINER(float, floats)
    REGISTER_ARG_SETTER_CONTAINER(double, floats)

    // Configure an operator type, inputs and outputs
    static void set_op(caffe2::OperatorDef &op,
		       const std::string &type,
		       const std::vector<std::string> &inputs,
		       const std::vector<std::string> &outputs) {
      op.set_type(type);
      for (const std::string &input : inputs) op.add_input(input);
      for (const std::string &output : outputs) op.add_output(output);
    }

    caffe2::OperatorDef &add_op(caffe2::NetDef &net, const caffe2::OperatorDef &op) {
      caffe2::OperatorDef &copy = *net.add_op();
      copy.CopyFrom(op);
      return copy;
    }

    // For each device, an operator is created, initialized with init_op, configured for the
    // current device, updated with fix_op, and then added to the net
    static void add_op_for_each_device(ScopedNet &net,
				       const OpModifier &init_op,
				       const OpModifier &fix_op
				       =[](caffe2::OperatorDef&){}) {

      const std::vector<caffe2::DeviceOption> *devices = &net._devices;

#ifndef CPU_ONLY
      /* Handle the _force_device tag */
      std::vector<caffe2::DeviceOption> buffer(1);
      if (net._force_device >= 0) {
	buffer[0].set_device_type(caffe2::CUDA);
	buffer[0].set_cuda_gpu_id(net._force_device);
	devices = &buffer;
      }
#endif

      for (const caffe2::DeviceOption &option : *devices) {
	caffe2::OperatorDef &op = *net._net.get().add_op();
	init_op(op);
	op.mutable_device_option()->CopyFrom(option);

#ifndef CPU_ONLY
	if (option.device_type() == caffe2::CUDA) {
	  std::string prefix = device_id_to_prefix(option.cuda_gpu_id());
	  /* Handle _rename_* tags */
	  if (net._rename_inputs) {
	    for (std::string &name : *op.mutable_input()) {
	      name = prefix + name;
	    }
	  }
	  if (net._rename_outputs) {
	    for (std::string &name : *op.mutable_output()) {
	      name = prefix + name;
	    }
	  }
	}
#endif
	fix_op(op);
	net._op_modifier(op);
      }
    }

    void add_op(ScopedNet &net, const caffe2::OperatorDef &op) {
      add_op_for_each_device(net, [&](caffe2::OperatorDef &copy) {
	  copy.CopyFrom(op);
	});
    }

#define ADD_ARG_VALUE(arg, value) add_arg(op, #arg, value)
#define ADD_ARG(arg) ADD_ARG_VALUE(arg, arg)
#define NO_ARG (void)op
#define BLOBS(...) std::vector<std::string>({__VA_ARGS__})
#define NAME(n) const std::string &n
#define VECTOR(t, n) const std::vector<t> &n
    //Define for NetDef, ScopedNet and OperatorDef
#define REGISTER_OP(name, input, output, args, proto...)		\
    caffe2::OperatorDef &name(caffe2::NetDef &net, proto) {		\
      caffe2::OperatorDef &op = *net.add_op();				\
      set_op(op, #name, input, output);					\
      args;								\
      return op;							\
    }									\
    void name(ScopedNet &net, proto) {					\
      add_op_for_each_device(net, [&](caffe2::OperatorDef &op) {	\
	  set_op(op, #name, input, output);				\
	}, [&](caffe2::OperatorDef &op) { args; });			\
    }									\
    void name(caffe2::OperatorDef &op, proto) {				\
      set_op(op, #name, input, output);					\
      args;								\
    }

    // input blob == output blob
#define REGISTER_SIMPLE_OP(name)					\
    REGISTER_OP(name, BLOBS(blob), BLOBS(blob), NO_ARG, NAME(blob))

    // N input and one output
#define REGISTER_SIMPLE_OP_1I1O(name)				\
  REGISTER_OP(name, BLOBS(input), BLOBS(output), NO_ARG,	\
	      NAME(input), NAME(output))
#define REGISTER_SIMPLE_OP_2I1O(name)					\
  REGISTER_OP(name, BLOBS(input1, input2), BLOBS(output), NO_ARG,	\
	      NAME(input1), NAME(input2), NAME(output))
#define REGISTER_SIMPLE_OP_3I1O(name)						\
  REGISTER_OP(name, BLOBS(input1, input2, input3), BLOBS(output), NO_ARG,	\
	      NAME(input1), NAME(input2), NAME(input3), NAME(output));

    // one input, one output and one argument
#define REGISTER_SIMPLE_OP_1I1O1A(name, type, arg)			\
    REGISTER_OP(name, BLOBS(input), BLOBS(output), ADD_ARG(arg),	\
		NAME(input), NAME(output), type arg)

    //    one output and a shape
    // or one output and an input (used as a shape)
#define REGISTER_SIMPLE_OP_FILLER(name)				\
    REGISTER_SIMPLE_OP_1I1O(name)				\
    REGISTER_OP(name, BLOBS(), BLOBS(output), ADD_ARG(shape),	\
		NAME(output), VECTOR(int, shape))

    /*
     *  Operators declaration
     */

    // Database
    REGISTER_OP(CreateDB, BLOBS(), BLOBS(reader),
		ADD_ARG(db); ADD_ARG_VALUE(db_type, "lmdb"),
		NAME(reader), NAME(db));
    REGISTER_OP(TensorProtosDBInput, BLOBS(reader), BLOBS(data, label), ADD_ARG(batch_size),
		NAME(reader), NAME(data), NAME(label), int batch_size);
    REGISTER_SIMPLE_OP_1I1O(NHWC2NCHW);

    // Basic
    REGISTER_SIMPLE_OP_1I1O(Copy);
    REGISTER_SIMPLE_OP_1I1O1A(Scale, float, scale);

    // Sum and Optimize
    REGISTER_OP(Sum, inputs, BLOBS(output), NO_ARG, VECTOR(std::string, inputs), NAME(output));
    REGISTER_OP(WeightedSum, inputs, BLOBS(output), NO_ARG,
		VECTOR(std::string, inputs), NAME(output));
    REGISTER_OP(MomentumSGDUpdate,
		BLOBS(gradient, momentum, rate, param), BLOBS(gradient, momentum, param), NO_ARG,
		NAME(param), NAME(momentum), NAME(gradient), NAME(rate));
    REGISTER_OP(Adagrad, BLOBS(param, momentum, gradient, rate), BLOBS(param, momentum), NO_ARG,
		NAME(param), NAME(momentum), NAME(gradient), NAME(rate));
    REGISTER_OP(Adam, BLOBS(param, momentum1, momentum2, gradient, rate, iter),
		BLOBS(param, momentum1, momentum2), NO_ARG,
		NAME(param), NAME(momentum1), NAME(momentum2),
		NAME(gradient), NAME(rate), NAME(iter))
    REGISTER_OP(RmsProp,
		BLOBS(gradient, mean_square, momentum, rate),
		BLOBS(gradient, mean_square, momentum),
		ADD_ARG_VALUE(decay, 0.9f);
		ADD_ARG_VALUE(momentum, 0.8f);
		ADD_ARG_VALUE(epsilon, 1e-5f),
		NAME(gradient), NAME(mean_square), NAME(momentum), NAME(rate));

    // Fill
    //ConstantFill
    REGISTER_SIMPLE_OP_1I1O1A(ConstantFill, float, value);
    REGISTER_OP(ConstantFill, BLOBS(), BLOBS(output),
		ADD_ARG(shape); ADD_ARG(value),
		NAME(output), VECTOR(int, shape), float value);
    REGISTER_OP(ConstantFill, BLOBS(), BLOBS(output),
		ADD_ARG(shape); ADD_ARG(value);
		ADD_ARG_VALUE(dtype, caffe2::TensorProto_DataType_INT64),
		NAME(output), VECTOR(int, shape), int value);
    //GivenTensorFill
    REGISTER_OP(GivenTensorFill, BLOBS(), BLOBS(output), ADD_ARG(shape); ADD_ARG(values),
		NAME(output), VECTOR(int, shape), VECTOR(float, values));
    //Classic fill
    REGISTER_SIMPLE_OP_FILLER(XavierFill);
    REGISTER_SIMPLE_OP_FILLER(GaussianFill);
    REGISTER_SIMPLE_OP_FILLER(MSRAFill);
    REGISTER_SIMPLE_OP_FILLER(RangeFill);
    REGISTER_SIMPLE_OP_1I1O(LengthsRangeFill);
    //XXXFill
    // DiagonalFill
    // UniformFill
    // UniformIntFill
    // UniqueUniformFill

    // Train
    REGISTER_SIMPLE_OP(Iter);
    REGISTER_SIMPLE_OP(StopGradient);
    REGISTER_OP(LearningRate, BLOBS(iter), BLOBS(rate),
		ADD_ARG(policy); ADD_ARG(base_lr); ADD_ARG(stepsize); ADD_ARG(gamma),
		NAME(iter), NAME(rate), NAME(policy), float base_lr, int stepsize, float gamma);

    // Test
    REGISTER_SIMPLE_OP_2I1O(LabelCrossEntropy);
    REGISTER_SIMPLE_OP_1I1O(AveragedLoss);
    REGISTER_SIMPLE_OP_2I1O(Accuracy);
    REGISTER_SIMPLE_OP_1I1O(Softmax);

    // Misc
    REGISTER_SIMPLE_OP_3I1O(FC);
    REGISTER_OP(Conv, BLOBS(input, w, b), BLOBS(output), 
		ADD_ARG(stride); ADD_ARG(pad); ADD_ARG(kernel); ADD_ARG_VALUE(order, "NCHW"),
		NAME(input), NAME(w), NAME(b), NAME(output), int stride, int pad, int kernel);
    REGISTER_OP(MaxPool, BLOBS(input), BLOBS(output),
		ADD_ARG(stride); ADD_ARG(pad); ADD_ARG(kernel);
		ADD_ARG_VALUE(order, "NCHW"); ADD_ARG_VALUE(legacy_pad, 3),
		NAME(input), NAME(output), int stride, int pad, int kernel);

    /*
     *  Operators Grouping
     */

    void insert_db_input_operator(ScopedNet &net, const caffe2::OperatorDef &dbinput) {

      // The same DBReader is shared on every device
      Caffe2NetTools::ScopeKeeper sk(net);
      net._net.get().add_external_input(dbinput.input(0));
      net._rename_inputs = false;

#ifndef CPU_ONLY
      if (net._devices[0].device_type() == caffe2::CUDA) {
	for (const caffe2::DeviceOption &option : net._devices) {
	  net._force_device = option.cuda_gpu_id();
	  add_op(net, dbinput);
	}
      } else
#endif
	add_op(net, dbinput);
    }

    void insert_learning_operators(ScopedNet &net, ScopedNet &init,
				   int iter, const std::string &policy,
				   float base_lr, int stepsize, float gamma) {
      Caffe2NetTools::ScopeKeeper sk(init);
      // Forcing the device (iter blobs must be run on CPU)
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::CPU);
      init._devices = {option};
      std::string main_iter;

      for (size_t i = 0; i < net._devices.size(); ++i) {
	std::string prefixed_iter = get_device_prefix(net._devices[i]) + blob_iter;
	// Broadcasting
	if (i) {
	  Copy(init, main_iter, prefixed_iter);
	} else {
	  main_iter = prefixed_iter;
	  ConstantFill(init, main_iter, {1}, iter);
	}
      }

      add_external_input(net, blob_iter);
      Iter(net, blob_iter);
      LearningRate(net, blob_iter, blob_lr, policy, base_lr, stepsize, gamma);
    }

    void insert_loss_operators(ScopedNet &net,
			       const std::string &prediction,
			       const std::string &label) {
      LabelCrossEntropy(net, prediction, label, blob_xent);
      AveragedLoss(net, blob_xent, blob_loss);
      Scale(net, blob_loss, blob_loss_scale, 1.f / net._devices.size());
      ConstantFill(net, blob_loss_scale, blob_loss_scale + gradient_suffix, 1.0);
    }

    // Add all the operators on the main device and copy the outputs on the other devices
    static void copy_and_broadcast_operators(ScopedNet &net,
					     const std::vector<const caffe2::OperatorDef *> &ops) {
#ifndef CPU_ONLY
      std::vector<std::string> sync;
      const caffe2::DeviceOption &main_device = net._devices[0];
      bool is_gpu = main_device.device_type() == caffe2::CUDA;
      {
	ScopeKeeper sk(net);
	if (is_gpu) {
	  net._force_device = main_device.cuda_gpu_id();
	}
#endif
	for (const caffe2::OperatorDef *op : ops) {
	  add_op(net, *op);
#ifndef CPU_ONLY
	  if (is_gpu) {
	    for (const std::string &output : op->output()) {
	      sync.push_back(output);
	    }
	  }
	}
      }
      for (const std::string &blob : sync) {
	broadcast(net, blob);
#endif
      }
    }

    void copy_and_broadcast_operator(ScopedNet &net, const caffe2::OperatorDef &op) {
      copy_and_broadcast_operators(net, {&op});
    }

    void copy_and_broadcast_operators(ScopedNet &dest, const caffe2::NetDef &src) {
      std::vector<const caffe2::OperatorDef *> ops;
      for (const caffe2::OperatorDef &op : src.op()) {
	ops.push_back(&op);
      }
      copy_and_broadcast_operators(dest, ops);
    }

  }
}
