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

    // Fetch the 'idx'th output of an operator
    // Create the requested filler for it, and place it in the 'fillers' map
    static caffe2::OperatorDef *add_filler(const caffe2::OperatorDef &op,
					   std::map<std::string, caffe2::OperatorDef> &fillers,
					   const std::string &ftype, int input_idx) {
      caffe2::OperatorDef *filler = NULL;
      if (input_idx < op.input().size()) {
	const std::string &output = op.input(input_idx);
	filler = &fillers[output];
	filler->set_type(ftype);
	filler->add_output(output);
      }
      return filler;
    }

    // Same as above, but also adds a constant value to the filler
    static caffe2::OperatorDef *add_filler(const caffe2::OperatorDef &op,
					   std::map<std::string, caffe2::OperatorDef> &fillers,
					   const std::string &ftype, int input_idx, float value) {
      caffe2::OperatorDef *filler = add_filler(op, fillers, ftype, input_idx);
      if (filler) {
	add_arg(*filler, "value", value);
      }
      return filler;
    }

    // Filler functions take an operator, and create a filler for each input referenced
    // as being a parameter of the net. Thoses inputs and their newly created fillers are storeds
    // in the 'fillers' map.
    using Filler = std::function<void(const caffe2::OperatorDef &, // Input
				      std::map<std::string,caffe2::OperatorDef> & // Output
				      )>;

    // Just to make it more readable
#define FILLER_LAMBDA [](const caffe2::OperatorDef &o, std::map<std::string,caffe2::OperatorDef> &f)
#define ADD_FILLER(type, args...) add_filler(o, f, #type "Fill", args)

    // Map linking operator types to their filler lambda
    static const std::map<std::string, Filler> op_fillers ({
	{
	  "Conv", FILLER_LAMBDA {
	    ADD_FILLER(Xavier, 1);
	    ADD_FILLER(Constant, 2);
	  }
	}, {
	  "FC", FILLER_LAMBDA {
	    ADD_FILLER(Xavier, 1);
	    ADD_FILLER(Constant, 2);
	  }
	}, {
	  "SpatialBN", FILLER_LAMBDA {
	    ADD_FILLER(Constant, 1, 1.0f);
	    ADD_FILLER(Constant, 2, 0.0f);
	    ADD_FILLER(Constant, 3, 0.0f);
	    ADD_FILLER(Constant, 4, 1.0f);
	  }
	}
      });

#undef FILLER_LAMBDA
#undef ADD_FILLER

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

    void reset_fillers(const caffe2::NetDef &net, caffe2::NetDef &init) {
      std::map<std::string, caffe2::OperatorDef> fillers;
      collect_filler_types(net, fillers);
      apply_filler_types(init, fillers);
    }

    /*
     *  Optimizers
     */

    // Optimizers all need the same variables, and their workflow are similar.
    // Common behavior was put in a class to prevent code duplication
    class AbstractOptimizer {

      // Keep an access to the nets
      const ModelContext &_context;
      caffe2::NetDef &_netdef;
      caffe2::NetDef &_initdef;
      std::map<std::string, std::vector<int> > _shapes;

    protected:

      // Members the childs may need
      ScopedNet _net;
      std::string _param;
      std::string _grad, _moment, _meansq; // Blob names

      // Create a new ConstantFill :
      //  - On the main device of the init net
      //  - Broadcasted on every device
      //  - Tagged as 'external input' on the main net
      void broadcast_external_constantfill(const std::string &name,
					   const std::vector<int> &shape,
					   float value) {
	caffe2::OperatorDef fill;
	ConstantFill(fill, name, shape, value);
	copy_and_broadcast_operator(_context, _initdef, fill);
	add_external_input(_net, name);
      }

      // Same as above, but using current parameter size, and a fill of 0
      void default_fillers(const std::vector<std::string> &names) {
	for (const std::string &name : names) {
	  broadcast_external_constantfill(name, _shapes[_param], 0);
	}
      }

      // Functions supposed to be overloaded
      virtual void init() {} // Once per net (optional)
      virtual void optimize() = 0; // Once per parameter

    public:

      AbstractOptimizer(const ModelContext &context,
			caffe2::NetDef &netdef,
			caffe2::NetDef &initdef) :
	_context(context),
	_netdef(netdef),
	_initdef(initdef),
	_net(context.scope_net(netdef)) {
      }

      // Common code
      void run() {

	// Collect net's informations
	std::vector<std::string> params;
	std::vector<std::string> computed;
	std::string main_prefix = _context.get_prefix(0);
	collect_params(_netdef, params, computed, main_prefix);
	_shapes.clear();
	collect_blob_shapes(_netdef, _shapes, main_prefix);

	// Call child's "init" function
	init();

	for (const std::string &param : params) {

	  // Set the current 'param' members
	  _param = param;
	  _grad = param + gradient_suffix;
	  _moment = param + momentum_suffix;
	  _meansq = param + mean_square_suffix;

	  // Call child's "optimize" function
	  optimize();
	}
      }

      // Transform a child into a callback
      template <class C>
      static Optimizer callback() {
	return [](const ModelContext &context, caffe2::NetDef &netdef, caffe2::NetDef &initdef) {
	  C(context, netdef, initdef).run();
	};
      }
    };

    class sgd : public AbstractOptimizer {
      using AbstractOptimizer::AbstractOptimizer;
      virtual void init() {
	broadcast_external_constantfill(blob_one, std::vector<int>({1}), 1);
      }
      virtual void optimize() {
	WeightedSum(_net, { _param, blob_one, _grad, blob_lr }, _param);
      }
    };

    class momentum : public AbstractOptimizer {
      using AbstractOptimizer::AbstractOptimizer;
      virtual void optimize() {
	default_fillers({_moment});
	MomentumSGDUpdate(_net, _param, _moment, _grad, blob_lr);
      }
    };

    class adagrad : public AbstractOptimizer {
      using AbstractOptimizer::AbstractOptimizer;
      virtual void optimize() {
	default_fillers({_moment});
	Adagrad(_net, _param, _moment, _grad, blob_lr);
      }
    };

    class adam : public AbstractOptimizer {
      using AbstractOptimizer::AbstractOptimizer;
      virtual void optimize() {
	std::string moment1(_moment + "_1");
	std::string moment2(_moment + "_2");
	default_fillers({moment1, moment2});
	Adam(_net, _param, moment1, moment2, _grad, blob_lr, blob_iter);
      }
    };

    class rmsprop : public AbstractOptimizer {
      using AbstractOptimizer::AbstractOptimizer;
      virtual void optimize() {
	default_fillers({_moment, _meansq});
	RmsProp(_net, _grad, _meansq, _moment, blob_lr);
	Sum(_net, { _param, _grad }, _param);
      }
    };

#define REGISTER_OPTIMIZER(name) { #name, AbstractOptimizer::callback<name>() }
    const std::map<std::string, Optimizer> optimizers = {
      REGISTER_OPTIMIZER(sgd),
      REGISTER_OPTIMIZER(momentum),
      REGISTER_OPTIMIZER(adagrad),
      REGISTER_OPTIMIZER(adam),
      REGISTER_OPTIMIZER(rmsprop)
    };
#undef REGISTER_OPTIMIZER

    const Optimizer &get_optimizer(const std::string &name) {
      const auto it = optimizers.find(name);
      if (it == optimizers.end()) {
	CAFFE_THROW("Optimizer '", name, "' is not supported.");
      }
      return it->second;
    }

  }
}
