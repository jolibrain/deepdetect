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
     *  Parameter fillers
     */

#define FILLER_FUNCTION_NAME(name) add_##name##_fillers

    // Filler functions take an operator, and create a filler for each input referenced
    // as being a parameter of the net. Thoses inputs and their newly created fillers are storeds
    // in the 'fillers' map.
#define REGISTER_FILLER_FUNCTION(name, code...)					\
    static void FILLER_FUNCTION_NAME(name)(const caffe2::OperatorDef &op,	\
					   std::map<std::string,		\
					   caffe2::OperatorDef> &fillers) {	\
      code;									\
    }

    // Fetch the 'idx'th output of an operator
    // Create the requested filler for it, and place it in the 'fillers' map
#define ADD_FILLER(name, idx, code...)				\
    if (idx < op.input().size()) {				\
      const std::string &output = op.input(idx);		\
      caffe2::OperatorDef &filler = fillers[output];		\
      filler.set_type(#name "Fill");				\
      filler.add_output(output);				\
      code;							\
    }

    // Add a constant value to a filler
#define ADD_FILLER_VALUE(name, idx, value)			\
    ADD_FILLER(name, idx, add_arg(filler, "value", value))

    //XXX Find when other filling functions may be needed
    REGISTER_FILLER_FUNCTION(Conv, ADD_FILLER(Xavier, 1) ADD_FILLER(Constant, 2))
    REGISTER_FILLER_FUNCTION(FC, ADD_FILLER(Xavier, 1) ADD_FILLER(Constant, 2))
    REGISTER_FILLER_FUNCTION(SpatialBN,
			     ADD_FILLER_VALUE(Constant, 1, 1.0f)
			     ADD_FILLER_VALUE(Constant, 2, 0.0f)
			     ADD_FILLER_VALUE(Constant, 3, 0.0f)
			     ADD_FILLER_VALUE(Constant, 4, 1.0f))

#define REGISTER_OP_FILLER(name) { #name, FILLER_FUNCTION_NAME(name) }

    // Map linking operator types to their filler function
    static const std::map<std::string, void(*)(const caffe2::OperatorDef&,
					       std::map<std::string,caffe2::OperatorDef>&)>
    op_fillers({
	REGISTER_OP_FILLER(Conv),
	REGISTER_OP_FILLER(FC),
	REGISTER_OP_FILLER(SpatialBN),
    });

    // Create fillers for every parameters of the net
    static void collect_filler_types(const caffe2::NetDef &net,
				     std::map<std::string, caffe2::OperatorDef> &fillers) {
      for (const caffe2::OperatorDef &op : net.op()) {
	auto it = op_fillers.find(op.type());
	if (it != op_fillers.end()) {
	  it->second(op, fillers);
	}
      }
    }

    // Replace the net current fillers (usually GivenTensorFills) with the ones
    // referenced in the 'fillers' map
    static void apply_filler_types(caffe2::NetDef &net,
				   const std::map<std::string, caffe2::OperatorDef> &fillers) {
      for (caffe2::OperatorDef &op : *net.mutable_op()) {
	auto it = fillers.find(op.output(0));
	if (it == fillers.end()) {
	  continue;
	}
	caffe2::OperatorDef copy = it->second;
	for (const caffe2::Argument &arg : op.arg()) {
	  if (arg.name() == "shape") {
	    copy.add_arg()->CopyFrom(arg);
	    break;
	  }
	}
	op.Swap(&copy);
      }
    }

    //XXX Redundant with the gradients.collect_params function
    /*
     * \brief browse the net and list shapes of blobs initialized by filling operators
     * @param net net to browse
     * @param shapes used to store blobs name and shape
     * @param prefix filter out blob names that don't start with this prefix
     * @param remove_prefix whether the given prefix must be removed from the blob name
     */
    static void collect_blob_shapes(const caffe2::NetDef &net,
				    std::map<std::string, std::vector<int> > &shapes,
				    const std::string &prefix = "",
				    bool remove_prefix = true) {
      for (const caffe2::OperatorDef &op : net.op()) {
	auto it = non_trainable_ops.find(op.type());
	if (it == non_trainable_ops.end() || !it->second || // Not a filling operator
	    op.input().size() || op.output().size() != 1 || // Shape not stored in arguments
	    op.output(0).find(prefix)) { // Does not start with the given prefix
	  continue;
	}
	std::string output = op.output(0).substr(remove_prefix * prefix.size(), -1);
	for (const caffe2::Argument &arg : op.arg()) {
	  if (arg.name() == "shape") {
	    shapes[output].assign(arg.ints().begin(), arg.ints().end());
	    break;
	  }
	}
      }
    }

    static void collect_blob_shapes(const ScopedNet &net,
				    std::map<std::string, std::vector<int> > &shapes,
				    bool remove_prefix = true) {
      collect_blob_shapes(net._net, shapes, get_device_prefix(net._devices[0]), remove_prefix);
    }

    void reset_fillers(const caffe2::NetDef &net, caffe2::NetDef &init) {
      std::map<std::string, caffe2::OperatorDef> fillers;
      collect_filler_types(net, fillers);
      apply_filler_types(init, fillers);
    }

    /*
     *  Optimizers
     */

#define REGISTER_OPTIMIZER(name, code) {			\
      #name, [](ScopedNet &net,					\
		ScopedNet &init) {				\
	std::vector<std::string> params;			\
	std::vector<std::string> computed;			\
	std::map<std::string, std::vector<int> > shapes;	\
	collect_params(net, params, computed);			\
	collect_blob_shapes(init, shapes);			\
	code;							\
      }								\
    }

#define BROADCAST_EXTERNAL_CONSTANTFILL(name, shape, value) {	\
      caffe2::OperatorDef name##_fill;				\
      ConstantFill(name##_fill, name, shape, value);		\
      copy_and_broadcast_operator(init, name##_fill);		\
      add_external_input(net, name);				\
    }

#define FOREACH_PARAM(code)				\
    for (const std::string &param : params) {		\
      std::string grad(param + gradient_suffix);	\
      std::string moment(param + momentum_suffix);	\
      std::string meansq(param + mean_square_suffix);	\
      const std::vector<int> &shape(shapes[param]);	\
      code;						\
    }

#define DEFAULT_FILLER(name) BROADCAST_EXTERNAL_CONSTANTFILL(name, shape, 0.f);

    const std::map<std::string, Optimizer> optimizers = {

      REGISTER_OPTIMIZER(sgd, {
	  BROADCAST_EXTERNAL_CONSTANTFILL(blob_one, std::vector<int>({1}), 1.f);
	  FOREACH_PARAM({
	      (void)shape;
	      WeightedSum(net, { param, blob_one, grad, blob_lr }, param);
	    });
	}),

      REGISTER_OPTIMIZER(momentum, FOREACH_PARAM({
	    DEFAULT_FILLER(moment);
	    MomentumSGDUpdate(net, param, moment, grad, blob_lr);
	  })),

      REGISTER_OPTIMIZER(adagrad, FOREACH_PARAM({
	    DEFAULT_FILLER(moment);
	    Adagrad(net, param, moment, grad, blob_lr);
	  })),

      REGISTER_OPTIMIZER(adam, FOREACH_PARAM({
	    std::string moment1(moment + "_1");
	    std::string moment2(moment + "_2");
	    DEFAULT_FILLER(moment1);
	    DEFAULT_FILLER(moment2);
	    Adam(net, param, moment1, moment2, grad, blob_lr, blob_iter);
	  })),

      REGISTER_OPTIMIZER(rmsprop, FOREACH_PARAM({
	    DEFAULT_FILLER(moment);
	    DEFAULT_FILLER(meansq);
	    RmsProp(net, grad, meansq, moment, blob_lr);
	    Sum(net, { param, grad }, param);
	  })),

    };

    const Optimizer &get_optimizer(const std::string &name) {
      const auto it = optimizers.find(name);
      if (it == optimizers.end()) {
	CAFFE_THROW("Optimizer '", name, "' is not supported.");
      }
      return it->second;
    }

  }
}
