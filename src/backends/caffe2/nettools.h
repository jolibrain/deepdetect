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
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/core/workspace.h>
#pragma GCC diagnostic pop

#include <caffe2/utils/proto_utils.h>

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Debug
     */

    //XXX Doesn't work on too heavy graphs
    void net_to_svg(const caffe2::NetDef &net, const std::string &path);

    /*
     *  Device management
     */

    /**
     * \breif alias to make things more readable
     */
    using OpModifier = std::function<void(caffe2::OperatorDef&)>;

    /**
     * \brief device-tagged net
     *
     *        When adding blobs or operators, each device will have its own version (e.g. below)
     *
     *        op {
     *          input: "gpu_1/data",	<- input renamed
     *          output: "gpu_1/data",	<- output renamed
     *          type: "StopGradient",
     *          device_option {		<- device option insered
     *            device_type: 1,
     *            cuda_gpu_id: 1
     *          }
     *        }
     *
     */
    class ScopedNet {

    public:

      std::reference_wrapper<caffe2::NetDef> _net;
      std::vector<caffe2::DeviceOption> _devices;

      bool _rename_inputs = true;
      bool _rename_outputs = true;

      // If set to a gpu id, devices vector will be ignored
      int _force_device = -1;

      // Called each time an operator is added to the net
      OpModifier _op_modifier = [](caffe2::OperatorDef&){};

      ScopedNet(caffe2::NetDef &n): _net(n) {}

    };

    /**
     * \breif backups ScopedNet tags and restore them upon destruction
     */
    class ScopeKeeper {
      ScopedNet &_ref;
      ScopedNet _backup;
    public:
      ScopeKeeper(ScopedNet &net): _ref(net), _backup(net) {}
      ~ScopeKeeper() { _ref = _backup; }
    };

#ifndef CPU_ONLY
    /**
     * \breif creates a prefix (e.g. gpu_1/)
     */
    std::string device_id_to_prefix(int id);
#endif
    /**
     * \breif uses the device type and id to create a prefix (e.g. gpu_1/)
     */
    std::string get_device_prefix(const caffe2::DeviceOption &option);

    /**
     * \brief tags the given input as external on each device
     */
    void add_external_input(ScopedNet &net, const std::string &input);

    /**
     * \brief tags the given output as external on each device
     */
    void add_external_output(ScopedNet &net, const std::string &output);

    /**
     * \brief sets set device_option of every operators
     */
    void set_net_device(caffe2::NetDef &net, const caffe2::DeviceOption &device);

    /*
     *	Protobuffer manipulation
     */

    /**
     * \brief removes the first argument with the given name (if any) from the operator
     */
    void del_arg(caffe2::OperatorDef &op, const std::string &name);

#define PROTOTYPE(type)						\
    /* Add an 'Argument' in an 'OperatorDef' */			\
    caffe2::Argument &add_arg(caffe2::OperatorDef &op,		\
			      const std::string& name,		\
			      const type &value);		\
    /* Same as del_arg followed by add_arg */			\
    caffe2::Argument &replace_arg(caffe2::OperatorDef &op,	\
				  const std::string& name,	\
				  const type &value);
    PROTOTYPE(int);
    PROTOTYPE(float);
    PROTOTYPE(std::string);
    PROTOTYPE(std::vector<int>);
    PROTOTYPE(std::vector<float>);
    PROTOTYPE(std::vector<double>);
#undef PROTOTYPE

    /**
     * \brief adds an operator into a net
     */
    caffe2::OperatorDef &add_op(caffe2::NetDef &net, const caffe2::OperatorDef &op);

    /**
     * \brief for each device, adds a copy of an operator into a NetDef
     */
    void add_op(ScopedNet &net, const caffe2::OperatorDef &op);

#define PROTOTYPE(name, args...)					\
    caffe2::OperatorDef &name(caffe2::NetDef&, args); /* adds a 'name' operator into a net */ \
    void name(ScopedNet&, args); /* for each device, adds a 'name' operator into a net */ \
    void name(caffe2::OperatorDef&, args); /* configures an operator as being a 'name' */

    // Database
    PROTOTYPE(CreateDB, const std::string &reader, const std::string &db);
    PROTOTYPE(TensorProtosDBInput,
	      const std::string &reader, const std::string &data, const std::string &label,
	      int batch_size);
    PROTOTYPE(NHWC2NCHW, const std::string &input, const std::string &output);

    // Basic
    PROTOTYPE(Copy, const std::string &input, const std::string &output);
    PROTOTYPE(Scale, const std::string &input, const std::string &output, float scale);

    // Sum and Optimize
    PROTOTYPE(Sum, const std::vector<std::string> &inputs, const std::string &output);
    PROTOTYPE(WeightedSum, const std::vector<std::string> &inputs, const std::string &output);
    PROTOTYPE(MomentumSGDUpdate, const std::string &param, const std::string &momentum,
	      const std::string &gradient, const std::string &rate);
    PROTOTYPE(Adagrad, const std::string &param, const std::string &momentum,
	      const std::string &gradient, const std::string &rate);
    PROTOTYPE(Adam, const std::string &param,
	      const std::string &momentum1, const std::string &momentum2,
	      const std::string &gradient, const std::string &rate, const std::string &iter);
    PROTOTYPE(RmsProp, const std::string &gradient, const std::string &mean_square,
	      const std::string &momentum, const std::string &rate);

    // Fill
    PROTOTYPE(ConstantFill, const std::string &input, const std::string &output, float value);
    PROTOTYPE(ConstantFill, const std::string &output, const std::vector<int> &shape, float value);
    PROTOTYPE(ConstantFill, const std::string &output, const std::vector<int> &shape, int value);
    PROTOTYPE(GivenTensorFill, const std::string &output,
	      const std::vector<int> &shape, const std::vector<float> &values);
    PROTOTYPE(XavierFill, const std::string &input, const std::string &output);
    PROTOTYPE(XavierFill, const std::string &output, const std::vector<int> &shape);
    PROTOTYPE(GaussianFill, const std::string &input, const std::string &output);
    PROTOTYPE(GaussianFill, const std::string &output, const std::vector<int> &shape);
    PROTOTYPE(MSRAFill, const std::string &input, const std::string &output);
    PROTOTYPE(MSRAFill, const std::string &output, const std::vector<int> &shape);
    PROTOTYPE(RangeFill, const std::string &input, const std::string &output);
    PROTOTYPE(RangeFill, const std::string &output, const std::vector<int> &shape);
    PROTOTYPE(LengthsRangeFill, const std::string &input, const std::string &output);

    // Train
    PROTOTYPE(Iter, const std::string &iter);
    PROTOTYPE(StopGradient, const std::string &blob);
    PROTOTYPE(LearningRate,
	      const std::string &iter, const std::string &rate, const std::string &policy,
	      float base_lr, int stepsize, float gamma);

    // Test
    PROTOTYPE(LabelCrossEntropy, const std::string &prediction, const std::string &label,
	      const std::string &output);
    PROTOTYPE(AveragedLoss, const std::string &xent, const std::string &loss);
    PROTOTYPE(Accuracy, const std::string &prediction, const std::string &label,
	      const std::string &output);
    PROTOTYPE(Softmax, const std::string &input, const std::string &output);

    // Misc
    PROTOTYPE(FC, const std::string &input, const std::string &weight, const std::string &bias,
	      const std::string &output);
    PROTOTYPE(Conv, const std::string &input, const std::string &weight, const std::string &bias,
	      const std::string &output, int stride, int pad, int kernel);
    PROTOTYPE(MaxPool, const std::string &input, const std::string &output,
	      int stride, int pad, int kernel);

#undef PROTOTYPE

    /*
     *  Operators Grouping
     */

    /**
     * \brief adds a tensor loader on each device, all sharing the same DBReader
     */
    void insert_db_input_operator(ScopedNet &net, const caffe2::OperatorDef &dbinput);

    /**
     * \brief adds operators related to the training on each device (iter, learning_rate, etc.)
     */
    void insert_learning_operators(ScopedNet &net, ScopedNet &init,
				   int iter, const std::string &policy,
				   float base_lr, int stepsize, float gamma);

    /**
     * \brief adds operators related to the loss computing on each device
     *        (cross entropy, averaged loss, iter, learning_rate, etc.)
     *        the loss will be scaled based on the number of device
     *        XXX Needs a label layer (supervised only)
     */
    void insert_loss_operators(ScopedNet &net,
			       const std::string &prediction,
			       const std::string &label);

    /**
     * \brief copies an operator on the main device and broadcasts the outputs on the others
     */
    void copy_and_broadcast_operator(ScopedNet &net, const caffe2::OperatorDef &op);

    /**
     * \brief copies every operators of the source on the main device
     *        and broadcasts the outputs on the others
     */
    void copy_and_broadcast_operators(ScopedNet &dest, const caffe2::NetDef &src);

    /*
     *  Workspace management
     */

    /**
     * \brief finds the blob of the given name and uses it to fill the tensor
     *        The device is used to know the blob's type, NOT to scope the blob name
     */
    bool extract_tensor(const caffe2::Workspace &workspace, const caffe2::DeviceOption &device,
			const std::string &name, caffe2::TensorCPU &tensor);

    /**
     * \brief finds the blob of the given name and fills it using the tensor
     *        The device is used to know the blob's type, NOT to scope the blob name
     */
    void insert_tensor(caffe2::Workspace &workspace, const caffe2::DeviceOption &device,
		       const std::string &name, const caffe2::TensorCPU &tensor);

    /**
     * \brief fetch the scaled losses of every devices and sums them
     */
    float extract_loss(const caffe2::Workspace &workspace,
		       const std::vector<caffe2::DeviceOption> &devices);

    /**
     * \brief fetch the current iteration
     */
    int extract_iter(const caffe2::Workspace &workspace, const caffe2::DeviceOption &device);

    /**
     * \brief serializes every blobs related to the training state (except parameters)
     *        and store them in a map
     */
    void extract_state(const caffe2::Workspace &workspace,
		       const caffe2::DeviceOption &device,
		       std::map<std::string, std::string> &blobs);

    /**
     * \brief creates an init net capable of setting the net parameters to their current value
     */
    void create_init_net(const caffe2::Workspace &workspace,
			 const caffe2::DeviceOption &device,
			 const caffe2::NetDef &net,
			 caffe2::NetDef &init);

    /**
     * \brief loads an iteration counter from a serialized blob
     */
    int load_iter(const std::string &path);

    /**
     * \brief loads a serialized blob and place it into the workspace
     */
    void load_blob(caffe2::Workspace &workspace,
		   const std::string &path,
		   const std::string &name);

    /**
     * \brief loads the serialized learning rate on each device
     */
    void load_lr(caffe2::Workspace &workspace,
		 const std::vector<caffe2::DeviceOption> &devices,
		 const std::string &path);

    /*
     *	Gradient management
     */

    /**
     * \brief checks if the operator have a gradient
     * @param op operator to check
     * @param trainable used to store the input blobs that will be part of the gradient
     * @param computed used to strore the other inputs
     * @return true if trainable, false otherwise
     */
    bool is_trainable(const caffe2::OperatorDef &op,
		      std::vector<std::string> *trainable,
		      std::vector<std::string> *computed);

    void add_gradient_ops(caffe2::NetDef &net);

    /*
     *  Gradient & Device
     */

    /**
     * \brief browses the net and lists external inputs that are used by trainable operators
     * @param net net to browse
     * @param params used to store input blobs that are used by the gradients
     * @param computed_params used to store the other input blobs
     * @param prefix filter out blob names that don't start with this prefix
     * @param remove_prefix whether the given prefix must be removed from the blob name
     */
    void collect_params(const caffe2::NetDef &net,
			std::vector<std::string> &params,
			std::vector<std::string> &computed_params,
			const std::string &prefix = "",
			bool remove_prefix = true);

    // Same as above, except the prefix is infered based on the first device's scope
    void collect_params(const ScopedNet &net,
			std::vector<std::string> &params,
			std::vector<std::string> &computed_params,
			bool remove_prefix = true);
#ifndef CPU_ONLY

    /**
     * \brief copies a blob from the first device to the others
     */
    void broadcast(ScopedNet &net, const std::string &blob);

    /**
     * \brief sums gradients over the devices and broadcasts the results
     */
    void reduce(ScopedNet &net);

#endif

    /*
     *  Parameter fillers
     */

    /**
     * \brief browses the net, creates a filler for each parameter,
     *        and uses them to update the init net.
     *        Only the fillers having an equivalent in the init net can be used,
     *        because their shape can't be infered.
     */
    void reset_fillers(const caffe2::NetDef &net, caffe2::NetDef &init);

    /*
     *  Optimizers
     */

    // Optimizers are strored as functions that take the net
    // and use the gradients to update the parameters
    //XXX Make epsilon, decay, etc. configurables
    using Optimizer = std::function<
      void
      (ScopedNet&,		// net
       ScopedNet&		// init_net
       )>;

    // List of registered optimizers : sgd, momentum, adagrad, adam, rmsprop
    //XXX Add other optimizers: SparseAdagrad, RowWiseSparseAdagrad, etc.
    const Optimizer &get_optimizer(const std::string &name);

    /*
     *  Others
     */

    /**
     * \brief truncates a net to keep only the operators and inputs needed to compute 'blob'
     *        This 'blob' is tagged as the external output of the net
     */
    void truncate_net(const caffe2::NetDef &net, caffe2::NetDef &out, const std::string &blob);

    // Same as above, except that the net is replaced by its truncated version
    void truncate_net(caffe2::NetDef &net, const std::string &blob);

  }
}
