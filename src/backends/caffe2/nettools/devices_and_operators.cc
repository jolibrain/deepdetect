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

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/core/operator.h>
#pragma GCC diagnostic pop

#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Simple tools
     */

    template<typename T, typename U>
    inline bool has_value(const T &container, const U &value) {
      return std::find(container.begin(), container.end(), value) != container.end();
    }

    bool has_input(const caffe2::OperatorDef &op, const std::string &name) {
      return has_value(op.input(), name);
    }
    bool has_output(const caffe2::OperatorDef &op, const std::string &name) {
      return has_value(op.output(), name);
    }

    int find_previous_update(const caffe2::NetDef &net, const std::string &name, int idx) {
      for (; idx >= 0; --idx) {
	if (has_output(net.op(idx), name)) {
	  return idx;
	}
      }
      CAFFE_THROW("Can't find '", name, "' first update");
    }

    /*
     *  Device management
     */

    /* ------------------------- Create prefix ------------------------- */

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

    /* ------------------------- Add input / output in loop ------------------------- */

    using ExtSetter = void (caffe2::NetDef::*)(const std::string &);

    inline void add_external(ScopedNet &net, const std::string &name,
			     const ExtSetter &setter, bool rename) {
      for (const caffe2::DeviceOption &option : net._devices) {
	std::string prefix = rename ? get_device_prefix(option) : "";
	(net._net.get().*setter)(prefix + name);
      }
    }

#define ADD_EXTERNAL(type)						\
    void add_external_##type(ScopedNet &net, const std::string &name) {	\
      add_external(net, name, &caffe2::NetDef::add_external_##type, net._rename_##type##s); \
    }									\
    void add_external_##type(caffe2::NetDef &net, const std::string &name) { \
      net.add_external_##type(name);					\
    }
    ADD_EXTERNAL(input)
    ADD_EXTERNAL(output)
#undef ADD_EXTERNAL

    /* ------------------------- Add op ------------------------- */

    inline bool is_cpu_only(const std::string &op) {
#ifdef CPU_ONLY
      return true;
#else
      return !has_value(caffe2::gDeviceTypeRegistry()->at(caffe2::CUDA)->Keys(), op);
#endif
    }

    // Returns true if successfull, false if CPU was forced
    static bool set_op_device(caffe2::OperatorDef &op, const caffe2::DeviceOption &device) {
      caffe2::DeviceOption &op_device = *op.mutable_device_option();
      if (is_cpu_only(op.type())) {
	op_device.set_device_type(caffe2::CPU);
	return device.device_type() == caffe2::CPU;
      }
      op_device.CopyFrom(device);
      return true;
    }

    static void add_op(caffe2::NetDef &net,
		       const caffe2::OperatorDef &op,
		       const caffe2::DeviceOption &device) {
      caffe2::OperatorDef new_op(op);
      if (set_op_device(new_op, device)) {
	net.add_op()->CopyFrom(new_op);
	return;
      }

      // Device unavailable

      // Convert & Rename inputs
      for (std::string &input : *new_op.mutable_input()) {
	caffe2::OperatorDef &to_cpu(EnsureCPUOutput(net, input, input + force_device_suffix));
	CAFFE_ENFORCE(set_op_device(to_cpu, device));
	input += force_device_suffix;
      }

      // Rename outputs
      for (std::string &output : *new_op.mutable_output()) {
	output += force_device_suffix;
      }

      net.add_op()->CopyFrom(new_op);

      // Convert outputs
      for (const std::string &output : op.output()) {
	caffe2::OperatorDef &from_cpu(CopyFromCPUInput(net, output + force_device_suffix, output));
	CAFFE_ENFORCE(set_op_device(from_cpu, device));
      }
    }

    /* ------------------------- Add ops in loop ------------------------- */

    void set_net_device(caffe2::NetDef &net, const caffe2::DeviceOption &device) {

      caffe2::NetDef tmp;
      for (const caffe2::OperatorDef &op : net.op()) {
	add_op(tmp, op, device);
      }
      net.mutable_op()->Swap(tmp.mutable_op());
    }

    /*
     *  Protobuffer manipulation
     */

    /* ------------------------- Add arg ------------------------- */

    template<typename T>
    using ArgSetter = void (caffe2::Argument::*)(T);

    // Single value
    template<typename T1, typename T2>
    inline void set_arg_value(caffe2::Argument &arg,
			      const T1 &value,
			      const ArgSetter<T2> &setter) {
      (arg.*setter)(value);
    }

    // Multiple values
    template<typename T1, typename T2>
    inline void set_arg_value(caffe2::Argument &arg,
			      const std::vector<T1> &values,
			      const ArgSetter<T2> &setter) {
      for (const T1 &value : values) {
	set_arg_value(arg, value, setter);
      }
    }

    // Create set and return
    template<typename T1, typename T2>
    inline caffe2::Argument &add_arg(caffe2::OperatorDef &op,
				     const std::string& name,
				     const T1 &value,
				     const ArgSetter<T2> &setter) {
      caffe2::Argument &arg = *op.add_arg();
      arg.set_name(name);
      set_arg_value(arg, value, setter);
      return arg;
    }

#define ADD_ARG(t1, t2, function)						\
    caffe2::Argument &add_arg(caffe2::OperatorDef &op,				\
			      const std::string& name,				\
			      t1 const &value) {				\
      return add_arg<t1, t2>(op, name, value, &caffe2::Argument::function);	\
    }
    //		Our type		Protobuf type		Name of the setter
    ADD_ARG(	int,			long int,		set_i)
    ADD_ARG(	bool,			long int,		set_i)
    ADD_ARG(	float,			float,			set_f)
    ADD_ARG(	char const*,		std::string const&,	set_s)
    ADD_ARG(	std::string,		std::string const&,	set_s)
    ADD_ARG(	std::vector<int>,	long int,		add_ints)
    ADD_ARG(	std::vector<float>,	float,			add_floats)
#undef ADD_ARG

    /* ------------------------- Add operator ------------------------- */

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

    // For each device, an operator is created, initialized and configured for the current device
    static void add_op_for_each_device(ScopedNet &net, const OpModifier &init) {

      const std::vector<caffe2::DeviceOption> *devices = &net._devices;

#ifndef CPU_ONLY
      // Handle the _force_device tag
      std::vector<caffe2::DeviceOption> buffer(1);
      if (net._force_device >= 0) {
	buffer[0].set_device_type(caffe2::CUDA);
	buffer[0].set_cuda_gpu_id(net._force_device);
	devices = &buffer;
      }
#endif

      for (const caffe2::DeviceOption &option : *devices) {
	caffe2::OperatorDef op;
	init(op);
#ifndef CPU_ONLY
	if (option.device_type() == caffe2::CUDA) {
	  std::string prefix = device_id_to_prefix(option.cuda_gpu_id());
	  // Handle _rename_* tags
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
	add_op(net._net, op, option);
      }
    }

    void add_op(ScopedNet &net, const caffe2::OperatorDef &op) {
      add_op_for_each_device(net, [&](caffe2::OperatorDef &copy) {
	  copy.CopyFrom(op);
	});
    }

    template<class Net>
    inline void add_ops_and_inputs(Net &dst, const caffe2::NetDef &src,
				   const std::vector<std::string> &ignore) {
      for (const caffe2::OperatorDef &op : src.op()) {
	add_op(dst, op);
      }
      for (const std::string &input : src.external_input()) {
	if (!has_value(ignore, input)) {
	  add_external_input(dst, input);
	}
      }
    }

    // Explicit declaration of the functions (easier to call, overload, and set default parameters)
#define ADD_OPS_AND_INPUTS(net)						\
    void add_ops_and_inputs(net &dst, const caffe2::NetDef &src,	\
			    const std::vector<std::string> &ignore) {	\
      add_ops_and_inputs<net>(dst, src, ignore);			\
    }
    ADD_OPS_AND_INPUTS(ScopedNet)
    ADD_OPS_AND_INPUTS(caffe2::NetDef)
#undef ADD_OPS_AND_INPUTS

    /* ------------------------- Declare operator ------------------------- */

    inline caffe2::OperatorDef &create_operator(caffe2::NetDef &net, const OpModifier &config) {
      caffe2::OperatorDef &op = *net.add_op();
      config(op);
      return op;
    }

    inline void create_operator(ScopedNet &net, const OpModifier &config) {
      add_op_for_each_device(net, config);
    }

    inline void create_operator(caffe2::OperatorDef &op, const OpModifier &config) {
      config(op);
    }

    // Define for NetDef, ScopedNet and OperatorDef
#define REGISTER_OP_FUNCTIONS(name, lambda, proto...)			\
    caffe2::OperatorDef &name(caffe2::NetDef &net, proto) {		\
      return create_operator(net, lambda);				\
    }									\
    void name(ScopedNet &net, proto) {					\
      return create_operator(net, lambda);				\
    }									\
    void name(caffe2::OperatorDef &op, proto) {				\
      return create_operator(op, lambda);				\
    }

    // OperatorDef arguments
#define ADD_ARG_VALUE(arg, value)	add_arg(op, #arg, value)
#define ADD_ARG(arg)			add_arg(op, #arg, arg)
#define NO_ARG

    // Blobs
#define BLOBS(...)			std::vector<std::string>({__VA_ARGS__})
#define INPUT				BLOBS
#define OUTPUT				BLOBS
#define NO_INPUT			BLOBS()
#define NO_OUTPUT			BLOBS()

    // Create a lambda with the requested configuration
#define REGISTER_OP(name, input, output, args, proto...)		\
    REGISTER_OP_FUNCTIONS(name, [&](caffe2::OperatorDef &op) {		\
	set_op(op, #name, input, output); args;				\
      }, proto)

    // input blob == output blob
#define REGISTER_SIMPLE_OP(name)					\
    REGISTER_OP(name,							\
		INPUT(blob),						\
		OUTPUT(blob),						\
		NO_ARG,							\
		const std::string &blob)

    // N input and one output
#define REGISTER_SIMPLE_OP_1I1O(name)					\
    REGISTER_OP(name,							\
		INPUT(input),						\
		OUTPUT(output),						\
		NO_ARG,							\
		const std::string &input,				\
		const std::string &output)
#define REGISTER_SIMPLE_OP_2I1O(name)					\
    REGISTER_OP(name,							\
		INPUT(input1, input2),					\
		OUTPUT(output),						\
		NO_ARG,							\
		const std::string &input1,				\
		const std::string &input2,				\
		const std::string &output)
#define REGISTER_SIMPLE_OP_3I1O(name)					\
    REGISTER_OP(name,							\
		INPUT(input1, input2, input3),				\
		OUTPUT(output),						\
		NO_ARG,							\
		const std::string &input1,				\
		const std::string &input2,				\
		const std::string &input3,				\
		const std::string &output)

    // one input, one output and one argument
#define REGISTER_SIMPLE_OP_1I1O1A(name, type, arg)			\
    REGISTER_OP(name,							\
		INPUT(input),						\
		OUTPUT(output),						\
		ADD_ARG(arg),						\
		const std::string &input,				\
		const std::string &output,				\
		type arg)

    //    one output and a shape
    // or one output and an input (used as a shape)
#define REGISTER_SIMPLE_OP_FILLER(name)				\
    REGISTER_SIMPLE_OP_1I1O(name)				\
    REGISTER_OP(name,						\
		NO_INPUT,					\
		OUTPUT(output),					\
		ADD_ARG(shape),					\
		const std::string &output,			\
		const std::vector<int> &shape)

    /*
     *  Operators declaration
     */

    // Database
    #define DB_TYPE "lmdb"
    REGISTER_OP(CreateDB,
		NO_INPUT,
		OUTPUT(reader),
		ADD_ARG(db);
		ADD_ARG_VALUE(db_type, DB_TYPE),
		const std::string &reader,
		const std::string &db)
    REGISTER_OP(TensorProtosDBInput,
		INPUT(reader),
		OUTPUT(data, label),
		ADD_ARG(batch_size),
		const std::string &reader,
		const std::string &data,
		const std::string &label,
		int batch_size)
    REGISTER_OP(ImageInput,
	        INPUT(reader),
		OUTPUT(data, label),
		ADD_ARG(batch_size);
		ADD_ARG(color);
		ADD_ARG_VALUE(minsize, size);
		ADD_ARG_VALUE(crop, size);
		ADD_ARG_VALUE(is_test, true);
		ADD_ARG(use_gpu_transform),
		const std::string &reader,
		const std::string &data,
		const std::string &label,
		int batch_size,
		int color,
		int size,
		bool use_gpu_transform)
    REGISTER_SIMPLE_OP_1I1O(NHWC2NCHW)

    // Basic
    REGISTER_OP(Sum,
		inputs,
		OUTPUT(output),
		NO_ARG,
		const std::vector<std::string> &inputs,
		const std::string &output)
    REGISTER_OP(Sub,
		INPUT(input1, input2),
		OUTPUT(output),
		ADD_ARG(broadcast); ADD_ARG(axis),
		const std::string &input1,
		const std::string &input2,
		const std::string &output,
		int broadcast, int axis)
    REGISTER_SIMPLE_OP_1I1O(Copy)
    REGISTER_SIMPLE_OP_1I1O(Alias)
    REGISTER_SIMPLE_OP_1I1O1A(Scale, float, scale)

    // Sum and Optimize
    REGISTER_OP(WeightedSum,
		inputs,
		OUTPUT(output),
		NO_ARG,
		const std::vector<std::string> &inputs,
		const std::string &output)
    REGISTER_OP(MomentumSGDUpdate,
		INPUT(gradient, momentum_blob, rate, param),
		OUTPUT(gradient, momentum_blob, param),
		//https://caffe2.ai/docs/operators-catalogue.html#momentumsgdupdate
		ADD_ARG_VALUE(nesterov, 1);
		ADD_ARG(momentum),
		const std::string &param,
		const std::string &momentum_blob,
		const std::string &gradient,
		const std::string &rate,
		float momentum)
    REGISTER_OP(Adagrad,
		INPUT(param, momentum, gradient, rate),
		OUTPUT(param, momentum),
		ADD_ARG_VALUE(epsilon, 1e-4f);
		ADD_ARG(decay),
		const std::string &param,
		const std::string &momentum,
		const std::string &gradient,
		const std::string &rate,
		float decay)
    REGISTER_OP(Adam,
		INPUT(param, momentum1, momentum2, gradient, rate, iter),
		OUTPUT(param, momentum1, momentum2),
		ADD_ARG_VALUE(epsilon, 1e-8f);
		ADD_ARG_VALUE(beta1, 0.9f);
		ADD_ARG_VALUE(beta2, 0.999f),
		const std::string &param,
		const std::string &momentum1,
		const std::string &momentum2,
		const std::string &gradient,
		const std::string &rate,
		const std::string &iter)
    REGISTER_OP(RmsProp,
		INPUT(gradient, mean_square, momentum_blob, rate),
		OUTPUT(gradient, mean_square, momentum_blob),
		ADD_ARG_VALUE(epsilon, 1e-5f);
		ADD_ARG(momentum); ADD_ARG(decay),
		const std::string &gradient,
		const std::string &mean_square,
		const std::string &momentum_blob,
		const std::string &rate,
		float momentum, float decay);

    // Fill
    //ConstantFill
    REGISTER_SIMPLE_OP_1I1O1A(ConstantFill, float, value)
    REGISTER_OP(ConstantFill,
		NO_INPUT,
		OUTPUT(output),
		ADD_ARG(shape); ADD_ARG(value),
		const std::string &output,
		const std::vector<int> &shape,
		float value)
    REGISTER_OP(ConstantFill,
		NO_INPUT,
		OUTPUT(output),
		ADD_ARG(shape); ADD_ARG(value);
		ADD_ARG_VALUE(dtype, caffe2::TensorProto_DataType_INT64),
		const std::string &output,
		const std::vector<int> &shape,
		int value)
    //GivenTensorFill
    REGISTER_OP(GivenTensorFill,
		NO_INPUT,
		OUTPUT(output),
		ADD_ARG(shape); ADD_ARG(values),
		const std::string &output,
		const std::vector<int> &shape,
		const std::vector<float> &values)
    //Classic fill
    REGISTER_SIMPLE_OP_FILLER(XavierFill)
    REGISTER_SIMPLE_OP_FILLER(GaussianFill)
    REGISTER_SIMPLE_OP_FILLER(MSRAFill)
    REGISTER_SIMPLE_OP_FILLER(RangeFill)
    REGISTER_SIMPLE_OP_1I1O(LengthsRangeFill)
    //XXXFill
    // DiagonalFill
    // UniformFill
    // UniformIntFill
    // UniqueUniformFill

    // Train
    REGISTER_SIMPLE_OP(Iter)
    REGISTER_SIMPLE_OP(StopGradient)
    REGISTER_OP(LearningRate,
		INPUT(iter),
		OUTPUT(rate),
		ADD_ARG(policy); ADD_ARG(base_lr);
		ADD_ARG(stepsize); ADD_ARG(max_iter);
		ADD_ARG(gamma); ADD_ARG(power),
		const std::string &iter,
		const std::string &rate,
		const std::string &policy,
		float base_lr,
		int stepsize,
		int max_iter,
		float gamma,
		float power)

    // Test
    REGISTER_SIMPLE_OP_2I1O(LabelCrossEntropy)
    REGISTER_SIMPLE_OP_1I1O(AveragedLoss)
    REGISTER_SIMPLE_OP_2I1O(Accuracy)
    REGISTER_SIMPLE_OP_1I1O(Softmax)

    // Misc
    REGISTER_SIMPLE_OP_1I1O(CopyFromCPUInput)
    REGISTER_SIMPLE_OP_1I1O(EnsureCPUOutput)
    REGISTER_SIMPLE_OP_3I1O(FC)
    REGISTER_OP(Conv,
		INPUT(input, w, b),
		OUTPUT(output),
		ADD_ARG(stride); ADD_ARG(pad); ADD_ARG(kernel); ADD_ARG_VALUE(order, "NCHW"),
		const std::string &input,
		const std::string &w,
		const std::string &b,
		const std::string &output,
		int stride, int pad, int kernel)
    REGISTER_OP(MaxPool,
		INPUT(input),
		OUTPUT(output),
		ADD_ARG(stride); ADD_ARG(pad); ADD_ARG(kernel);
		ADD_ARG_VALUE(order, "NCHW"); ADD_ARG_VALUE(legacy_pad, 3),
		const std::string &input,
		const std::string &output,
		int stride, int pad, int kernel)

#undef ADD_ARG_VALUE
#undef ADD_ARG
#undef NO_ARG

#undef BLOBS
#undef INPUT
#undef OUTPUT
#undef NO_INPUT
#undef NO_OUTPUT

#undef REGISTER_OP
#undef REGISTER_SIMPLE_OP
#undef REGISTER_SIMPLE_OP_1I1O
#undef REGISTER_SIMPLE_OP_2I1O
#undef REGISTER_SIMPLE_OP_3I1O
#undef REGISTER_SIMPLE_OP_1I1O1A
#undef REGISTER_SIMPLE_OP_FILLER

    /*
     *  Operators Grouping
     */

    /* ------------------------- Add operators ------------------------- */

    void insert_db_input_operator(const ModelContext &context, caffe2::NetDef &net_def,
				  const caffe2::OperatorDef &dbinput) {

      // The same DBReader is shared on every device
      net_def.add_external_input(dbinput.input(0));
      ScopedNet net = context.scope_net(net_def);
      net._rename_inputs = false;

#ifndef CPU_ONLY
      if (context._parallelized) {
	for (const caffe2::DeviceOption &option : context._devices) {
	  net._force_device = option.cuda_gpu_id();
	  add_op(net, dbinput);
	}
      } else
#endif
	add_op(net, dbinput);
    }

    void insert_learning_operators(const ModelContext &context,
				   caffe2::NetDef &net_def,
				   caffe2::NetDef &init_def,
				   const LROpModifier &lr_config) {

      ScopedNet net(context.scope_net(net_def));
      // Forcing the device (iter blobs must be run on CPU)
      ScopedNet init(init_def);
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::CPU);
      init._devices = {option};

      std::string main_iter;

      // Add iter blob
      for (size_t i = 0; i < context.device_count(); ++i) {
	std::string prefixed_iter = context.get_prefix(i) + blob_iter;
	// Broadcasting
	if (i) {
	  Copy(init, main_iter, prefixed_iter);
	} else {
	  main_iter = prefixed_iter;
	  ConstantFill(init, main_iter, {1}, context._loaded_iter);
	}
      }

      add_external_input(net, blob_iter);
      Iter(net, blob_iter);
      caffe2::OperatorDef lr;
      lr_config(lr, blob_iter, blob_lr);
      add_op(net, lr);
    }

    // Add all the operators on the main device and copy the outputs on the other devices
    static void copy_and_broadcast_operators(const ModelContext &context, caffe2::NetDef &net_def,
					     const std::vector<const caffe2::OperatorDef *> &ops) {
      ScopedNet net = context.scope_net(net_def);
#ifndef CPU_ONLY
      std::vector<std::string> sync;
      const caffe2::DeviceOption &main_device = context._devices[0];
      bool is_sync = main_device.device_type() == caffe2::CUDA && context._parallelized;

      if (is_sync) {
	net._force_device = main_device.cuda_gpu_id();
      }
#endif
      for (const caffe2::OperatorDef *op : ops) {
	add_op(net, *op);
#ifndef CPU_ONLY
	if (is_sync) {
	  for (const std::string &output : op->output()) {
	    sync.push_back(output);
	  }
	}
      }
      for (const std::string &blob : sync) {
	broadcast(net, blob);
#endif
      }
    }

    void copy_and_broadcast_operator(const ModelContext &context, caffe2::NetDef &net,
				     const caffe2::OperatorDef &op) {
      copy_and_broadcast_operators(context, net, {&op});
    }

    void copy_and_broadcast_operators(const ModelContext &context, caffe2::NetDef &net,
				      const caffe2::NetDef &src) {
      std::vector<const caffe2::OperatorDef *> ops;
      for (const caffe2::OperatorDef &op : src.op()) {
	ops.push_back(&op);
      }
      copy_and_broadcast_operators(context, net, ops);
    }

    /*
     *  Pre-computation
     */

    typedef std::function<void(caffe2::NetDef&,int)> EnsureIsBatchable;
    extern const std::map<std::string, EnsureIsBatchable> ensure_op_is_batchable;

    /*
     * Currently only the detectron models make us update the net
     * in order to make it batchable.
     *
     * By default the 'batch_splits' blob is not used, so we need to place it into the net.
     * See in https://github.com/pytorch/pytorch/blob/master/caffe2/operators :
     * box_with_nms_limit_op.cc and bbox_transform_op.cc
     *
     * Another problem occurs with CollectAndDistributeFpnRpnProposals
     * that do not keep the data grouped by batch item.
     * See in https://github.com/chichaj/deepdetect/blob/master/patches/caffe2 :
     * collect_proposals.patch
     */
    void ensure_is_batchable(caffe2::NetDef &net) {
      // Loop over the operators
      const auto &ops = net.op();
      for (int idx = ops.size(); idx-- > 0;) {
	auto it = ensure_op_is_batchable.find(ops[idx].type());
	if (it != ensure_op_is_batchable.end()) {
	  it->second(net, idx);
	}
      }
    }

    void make_input_batchable(caffe2::NetDef &net, int op_idx, int input_size,
			      const std::string &prev_type, int input_idx, int output_idx) {
      // Check if all the inputs are already present
      auto &inputs = *net.mutable_op(op_idx)->mutable_input();
      if (inputs.size() == input_size) {
	return;
      }
      CAFFE_ENFORCE(inputs.size() == input_size - 1);
      // Find a previous blob that can output the 'batch_splits' blob
      int prev_idx = find_previous_update(net, inputs[input_idx], op_idx - 1);
      const caffe2::OperatorDef &prev = net.op(prev_idx);
      CAFFE_ENFORCE(prev.type() == prev_type);
      // Add it the the input's list
      ensure_op_is_batchable.at(prev_type)(net, prev_idx);
      inputs.Add(std::string(prev.output(output_idx)));
    }

    void make_output_batchable(caffe2::NetDef &net, int op_idx, int output_size,
			       int output_idx) {
      // Check if all the outputs are already present
      auto &outputs = *net.mutable_op(op_idx)->mutable_output();
      if (outputs.size() == output_size) {
	return;
      }
      // Output a 'batch_splits' blob
      CAFFE_ENFORCE(outputs.size() == output_size - 1);
      outputs.Add(std::move(outputs[output_idx] + batch_splits_suffix));
    }

    static void ensure_collect_proposals_is_batchable(caffe2::NetDef &net, int idx) {
      caffe2::OperatorDef &op = *net.mutable_op(idx);
      for (caffe2::Argument &arg : *op.mutable_arg()) {
	if (arg.name() == "keep_grouped") {
	  arg.set_i(true);
	  return;
	}
      }
      add_arg(op, "keep_grouped", true);
    }

    static inline void ensure_bbox_transform_is_batchable(caffe2::NetDef &net, int idx) {
      make_output_batchable(net, idx, 2, 0);
    }

    static inline void ensure_box_nms_is_batchable(caffe2::NetDef &net, int idx) {
      make_input_batchable(net, idx, 3, "BBoxTransform", 1, 1);
      make_output_batchable(net, idx, 4, 0);
    }

    static inline void ensure_bbox_rois_is_batchable(caffe2::NetDef &net, int idx) {
      make_input_batchable(net, idx, 3, "BoxWithNMSLimit", 0, 3);
    }

    const std::map<std::string, EnsureIsBatchable> ensure_op_is_batchable({
	{ "CollectAndDistributeFpnRpnProposals", ensure_collect_proposals_is_batchable },
	{ "BBoxTransform", ensure_bbox_transform_is_batchable },
	{ "BoxWithNMSLimit", ensure_box_nms_is_batchable },
	{ "BBoxToRoi", ensure_bbox_rois_is_batchable },
    });

    /*
     *  Finalisation
     */

    static void remove_duplicate_casts(caffe2::NetDef &net) {

      caffe2::NetDef new_net;
      std::map<std::string, std::string> casted_blobs;
      std::map<std::string, std::string> rename_blobs;

      for (caffe2::OperatorDef &op : *net.mutable_op()) {

	// Rename inputs
	for (std::string &input : *op.mutable_input()) {
	  auto rename = rename_blobs.find(input);
	  if (rename != rename_blobs.end()) {
	    input = rename->second;
	  }
	}

	// Cast operators
	if (op.type() == "CopyFromCPUInput" || op.type() == "EnsureCPUOutput") {
	  const std::string &input = op.input(0);
	  const std::string &output = op.output(0);

	  // Reuse the previous casts
	  auto casted = casted_blobs.find(input);
	  if (casted != casted_blobs.end()) {
	    rename_blobs[output] = casted->second;
	    continue;
	  }

	  // Store the blobs name
	  casted_blobs[input] = output;
	  casted_blobs[output] = input;

	} else {

	  // Detect blob overrides
	  for (const std::string &output : op.output()) {
	    // Do not rename anymore
	    auto rename = rename_blobs.find(output);
	    if (rename != rename_blobs.end()) {
	      rename_blobs.erase(rename);
	    }
	    // Do not reuse casts
	    auto casted = casted_blobs.find(output);
	    if (casted != casted_blobs.end()) {
	      casted_blobs.erase(casted_blobs.find(casted->second));
	      casted_blobs.erase(casted);
	    }
	  }
	}

	// Apply modifications
	add_op(new_net, op);
      }
      net.mutable_op()->Swap(new_net.mutable_op());
    }

    // We consider that nets are not splitted during a CPU/CUDA conversion
    static void remove_useless_casts(caffe2::NetDef &net) {

      const auto &net_inputs(net.external_input());
      const std::set<std::string> external_inputs(net_inputs.begin(), net_inputs.end());

      for (int i = 0; i < net.op().size(); ++i) {
	caffe2::OperatorDef &op = *net.mutable_op(i);

	// Cast operators
	if (op.type() == "CopyFromCPUInput" || op.type() == "EnsureCPUOutput") {
	  const std::string &input = op.input(0);
	  const std::string &output = op.output(0);

	  bool internal = external_inputs.find(input) == external_inputs.end();
	  if (!internal) {
	    // The cast is not an external input, but may be an external output

	    // Check if the cast is used
	    for (int j = i; j < net.op().size(); ++j) {
	      if (has_input(net.op(j), output)) {
		internal = true;
		break;
	      }
	    }
	  }

	  // Transform into a simple alias
	  if (!internal) {
	    caffe2::OperatorDef new_op;
	    Alias(new_op, input, output);
	    op.Swap(&new_op);
	  }
	}
      }
    }

    void final_optimizations(caffe2::NetDef &net) {
      // Prevent useless casts between CPU/GPU tensors
      remove_duplicate_casts(net);
      remove_useless_casts(net);
    }

  }
}
