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

#ifndef CAFFE2NETTOOLS_H
#define CAFFE2NETTOOLS_H

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

    // Exports in three formats : <path>.pb, <path>.pbtxt and <path>.svg
    void dump_net(const caffe2::NetDef &net, const std::string &path);

    // Reads 'predict_net.pb' and 'init_net.pb' in the input folder, resets the weights,
    // and dumps the result as 'predict_net.pbtxt' and 'init_net.pbtxt' in the output folder
    void untrain_model(const std::string &input, const std::string &output);

    /*
     *  Device management
     */

    // Aliases to make things more readable
    using OpModifier = std::function<void(caffe2::OperatorDef&)>;
    using LROpModifier = std::function<void(caffe2::OperatorDef&, // LearningRate
					    const std::string&, // Iter blob
					    const std::string&)>; // LR blob

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
    void add_external_input(ScopedNet &net, const std::string &name);

    /**
     * \brief tags the given output as external on each device
     */
    void add_external_output(ScopedNet &net, const std::string &name);

    // Simple overload of the previous functions to be more uniform
    void add_external_input(caffe2::NetDef &net, const std::string &name);
    void add_external_output(caffe2::NetDef &net, const std::string &name);

    /**
     * \brief sets set device_option of every operators
     */
    void set_net_device(caffe2::NetDef &net, const caffe2::DeviceOption &device);

    /*
     *	Protobuffer manipulation
     */

    bool has_input(const caffe2::OperatorDef &op, const std::string &name);
    bool has_output(const caffe2::OperatorDef &op, const std::string &name);

    /**
     * \brief finds where the given blob was previously updataded
     * @param net net to search
     * @param name name of the blob
     * @param idx first operator index to check (the search is done by decreasing this index)
     * @return the index of the found operator
     */
    int find_previous_update(const caffe2::NetDef &net, const std::string &name, int idx);

    // Adds an 'Argument' in an 'OperatorDef'
#define PROTOTYPE(type)							\
    caffe2::Argument &add_arg(caffe2::OperatorDef &op, const std::string &name, const type &value);
    PROTOTYPE(int);
    PROTOTYPE(float);
    PROTOTYPE(std::string);
    PROTOTYPE(std::vector<int>);
    PROTOTYPE(std::vector<float>);
#undef PROTOTYPE

    /**
     * \brief adds an operator into a net
     */
    caffe2::OperatorDef &add_op(caffe2::NetDef &net, const caffe2::OperatorDef &op);

    /**
     * \brief for each device, adds a copy of an operator into a NetDef
     */
    void add_op(ScopedNet &net, const caffe2::OperatorDef &op);

    /**
     * \brief appends a net into another (adds operators and external inputs only)
     * @param dst destination net
     * @param dst source net
     * @param ignore external inputs to ignore (usefull if defined by the destination net)
     */
    void add_ops_and_inputs(caffe2::NetDef &dst, const caffe2::NetDef &src,
			    const std::vector<std::string> &ignore = {});
    // Same as above, except a copy is added for each devices
    void add_ops_and_inputs(ScopedNet &dst, const caffe2::NetDef &src,
			    const std::vector<std::string> &ignore = {});

    // Declare variants of the same function:
    //		- NetDef	adds a single operator into a net
    //		- ScopedNet	adds an operator for each device
    //		- OperatorDef	simply configures the operator
#define PROTOTYPE(name, args...)					\
    caffe2::OperatorDef &name(caffe2::NetDef&, args);			\
    void name(ScopedNet&, args);					\
    void name(caffe2::OperatorDef&, args);

    // Database
    PROTOTYPE(CreateDB, const std::string &reader, const std::string &db);
    PROTOTYPE(TensorProtosDBInput,
	      const std::string &reader, const std::string &data, const std::string &label,
	      int batch_size);
    PROTOTYPE(NHWC2NCHW, const std::string &input, const std::string &output);

    // Basic
    PROTOTYPE(Sum, const std::vector<std::string> &inputs, const std::string &output);
    PROTOTYPE(Sub, const std::string &input1, const std::string &input2, const std::string &output,
	      int broadcast, int axis);
    PROTOTYPE(Copy, const std::string &input, const std::string &output);
    PROTOTYPE(Alias, const std::string &input, const std::string &output);
    PROTOTYPE(Scale, const std::string &input, const std::string &output, float scale);

    // Sum and Optimize
    PROTOTYPE(WeightedSum, const std::vector<std::string> &inputs, const std::string &output);
    PROTOTYPE(MomentumSGDUpdate, const std::string &param, const std::string &momentum_blob,
	      const std::string &gradient, const std::string &rate, float momentum);
    PROTOTYPE(Adagrad, const std::string &param, const std::string &momentum,
	      const std::string &gradient, const std::string &rate, float decay);
    PROTOTYPE(Adam, const std::string &param,
	      const std::string &momentum1, const std::string &momentum2,
	      const std::string &gradient, const std::string &rate, const std::string &iter);
    PROTOTYPE(RmsProp, const std::string &gradient, const std::string &mean_square,
	      const std::string &momentum_blob, const std::string &rate,
	      float momentum, float decay);

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
	      float base_lr, int stepsize, int max_iter, float gamma, float power);

    // Test
    PROTOTYPE(LabelCrossEntropy, const std::string &prediction, const std::string &label,
	      const std::string &output);
    PROTOTYPE(AveragedLoss, const std::string &xent, const std::string &loss);
    PROTOTYPE(Accuracy, const std::string &prediction, const std::string &label,
	      const std::string &output);
    PROTOTYPE(Softmax, const std::string &input, const std::string &output);

    // Misc
    PROTOTYPE(CopyFromCPUInput, const std::string &input, const std::string &output);
    PROTOTYPE(EnsureCPUOutput, const std::string &input, const std::string &output);
    PROTOTYPE(FC, const std::string &input, const std::string &weight, const std::string &bias,
	      const std::string &output);
    PROTOTYPE(Conv, const std::string &input, const std::string &weight, const std::string &bias,
	      const std::string &output, int stride, int pad, int kernel);
    PROTOTYPE(MaxPool, const std::string &input, const std::string &output,
	      int stride, int pad, int kernel);

#undef PROTOTYPE

    /**
     * \brief transforms the net into a version supporting batching
     */
    void ensure_is_batchable(caffe2::NetDef &net);

    /**
     * \brief transforms the net into a quicker version
     *        /!\ Can remove and edit blobs and operators /!\
     */
    void final_optimizations(caffe2::NetDef &net);

    /*
     *  Workspace management
     */

    /**
     * \brief A workspace with its configuration
     */
    class ModelContext {
    public:

      // Workspaces cannot be std::move()'d or assigned
      // (see DISABLE_COPY_AND_ASSIGN in caffe2/core/workspace.h)
      // Hence the usage of a pointer.
      std::unique_ptr<caffe2::Workspace> _workspace =
	std::unique_ptr<caffe2::Workspace>(new caffe2::Workspace);
      std::vector<caffe2::DeviceOption> _devices;
      std::string _input_blob;
      std::vector<std::string> _output_blobs;
      int _nclasses = 0;

      //XXX Should be optionals / configurables in the future
      std::string _blob_label = "label";
      std::string _blob_im_info = "im_info";
      std::string _net_type = "dag";
      int _thread_per_device = 4;

      bool _parallelized; // Whether multiple devices are used
      int _loaded_iter; // Last iteration number that was loaded from the file system

      inline void reset_workspace() { _workspace.reset(new caffe2::Workspace); }
      inline size_t device_count() const { return _parallelized ? _devices.size() : 1; }
      inline std::string get_prefix(int device_idx) const {
	return _parallelized ? get_device_prefix(_devices[device_idx]) : "";
      }
      inline void create_input() {
	for (size_t i = 0; i < device_count(); ++i) {
	  _workspace->CreateBlob(get_prefix(i) + _input_blob);
	}
      }
      inline void create_net(caffe2::NetDef &net) const {
	net.set_type(_net_type);
	net.set_num_workers(_thread_per_device * device_count());
	final_optimizations(net);
	CAFFE_ENFORCE(_workspace->CreateNet(net));
      }

      // Enforce
      inline void run_net(const std::string &net) {
	CAFFE_ENFORCE(_workspace->RunNet(net));
      }
      inline void run_net_once(const caffe2::NetDef &net) {
	CAFFE_ENFORCE(_workspace->RunNetOnce(net));
      }

      /*
       *  Workspace initialization
       */

      /**
       * \brief creates a scoped net
       */
      ScopedNet scope_net(caffe2::NetDef &net) const;

      /**
       * \brief resets the list of devices to a single CPU
       */
      void reset_devices();

#ifndef CPU_ONLY
      /**
       * \brief resets the list of devices to multiple GPUs
       */
      void reset_devices(const std::vector<int> &gpu_ids);
#endif

      /**
       * \brief tries to find the blob of the given name on the given device,
       *        and to use the tensor to fill it
       */
      void insert_tensor(int device_idx, const std::string &name, const caffe2::TensorCPU &tensor);

      inline void reset_iter() { _loaded_iter = 0; }

      /**
       * \brief loads an iteration counter from a serialized blob
       *        (not into the workspace, just as an internal flag)
       */
      void load_iter(const std::string &path);

      /**
       * \brief loads a serialized blob and place it into the workspace
       */
      void load_blob(const std::string &path, const std::string &name);

      /**
       * \brief loads the serialized learning rate on each device
       */
      void load_lr(const std::string &path);

      /*
       *  Information insertion
       */

      /**
       * \brief adds a copy of the tensor on each device
       */
      void broadcast_tensor(const std::string &name, const caffe2::TensorCPU &tensor);

      /*
       *  Information extraction
       */

      /**
       * \brief tries to find the blob of the given name on the given device,
       *        and to use it to fill the tensor (true if successfull)
       */
      bool extract_tensor(int device_idx, const std::string &name, caffe2::TensorCPU &tensor) const;

      /**
       * \brief fetches the scaled losses of every devices and sums them
       */
      float extract_loss() const;

      /**
       * \brief fetches the current iteration
       */
      int extract_iter() const;

      /**
       * \brief fetches the labels and store them as float values (the vector size must be pre-set)
       */
      void extract_labels(std::vector<float> &labels) const;

      /**
       * \brief serializes every blobs related to the training state (except parameters)
       *        and store them in a map
       */
      void extract_state(std::map<std::string, std::string> &blobs) const;

      /**
       * \brief fetches the given layer and merge the results of each devices into a single batch.
       *        Note that the function does not append elements, but assign a value to them
       * @param results where to store the data
       * @param name name of the layer
       * @param sizes size of each element of the batch (split equally if empty)
       */
      void extract(std::vector<std::vector<float>> &results, const std::string &name,
		   const std::vector<size_t> &sizes={}) const;

      /*
       *  Network manipulation
       */

      //XXX Recreate the 'OptimizeGradientMemory' function to reuse blobs in the net

      /**
       * \brief creates an init net capable of setting the net parameters to their current value
       */
      void create_init_net(const caffe2::NetDef &net, caffe2::NetDef &init) const;

      /**
       * \bried appends a net's operators and inputs to another
       */
      void append_net(caffe2::NetDef &dst, const caffe2::NetDef &src) const;

      /**
       * \bried appends a net's operators and inputs to another (its 'main' input is ignored)
       *        then adds gradients for the new operators
       */
      void append_trainable_net(caffe2::NetDef &dst, const caffe2::NetDef &src) const;

    private:

      // Tools to generalize the 'extract_*' functions

      template <typename Result, typename Data>
      using Stockage = std::function<void(Result &, const Data *, size_t)>;

      // Extract from each device and return the total size
      size_t extract_tensors(const std::string &name,
			     std::vector<caffe2::TensorCPU> &tensors) const;

      // Merge the tensors and re-split into a single batch of items
      template <typename Result, typename Data>
      void split_tensors(std::vector<Result> &results,
			 std::vector<caffe2::TensorCPU> tensors,
			 const std::vector<size_t> &sizes,
			 const Stockage<Result, Data> &store) const;
      template <typename T>
      void split_tensors(std::vector<T> &results,
			 const std::vector<caffe2::TensorCPU> &tensors,
			 const std::vector<size_t> &sizes) const;
      template <typename T>
      void split_tensors(std::vector<std::vector<T>> &results,
			 const std::vector<caffe2::TensorCPU> &tensors,
			 const std::vector<size_t> &sizes) const;

      // Fecth the data then split it
      template <typename T>
      void extract_results(std::vector<T> &results,
			   const std::string &name,
			   size_t size=0) const;
      template <typename T>
      void extract_results(std::vector<T> &results,
			   const std::string &name,
			   const std::vector<size_t> &sizes) const;

      // Fetch, split and cast the data
      template <typename Result, typename Data, typename Size>
      void extract_and_cast_results(std::vector<Result> &results,
				    const std::string &name,
				    const Size &size) const;

    }; //! ModelContext

    /*
     *  Operators Grouping
     */

    /**
     * \brief adds a tensor loader on each device, all sharing the same DBReader
     */
    void insert_db_input_operator(const ModelContext &context, caffe2::NetDef &net_def,
				  const caffe2::OperatorDef &dbinput);

    /**
     * \brief adds operators related to the training on each device (iter, learning_rate, etc.)
     */
    void insert_learning_operators(const ModelContext &context,
				   caffe2::NetDef &net_def,
				   caffe2::NetDef &init_def,
				   const LROpModifier &lr_config);

    /**
     * \brief copies an operator on the main device and broadcasts the outputs on the others
     */
    void copy_and_broadcast_operator(const ModelContext &context, caffe2::NetDef &net,
				     const caffe2::OperatorDef &op);
    // Same as above but with every operator of the source
    void copy_and_broadcast_operators(const ModelContext &context, caffe2::NetDef &dest,
				      const caffe2::NetDef &src);

    /*
     *	Gradient management
     */

    /**
     * \brief checks if the operator have a gradient
     * @param op operator to check
     * @param trainable used to store the input blobs that will be part of the gradient
     * @param computed used to strore the other inputs
     * @param needed outputs that needs to be added to the operator to compute its gradient
     * @return true if trainable, false otherwise
     */
    bool is_trainable(const caffe2::OperatorDef &op,
		      std::set<std::string> *trainable,
		      std::set<std::string> *computed,
		      std::vector<std::string> *needed);

    void add_gradient_ops(caffe2::NetDef &net, const std::set<std::string> &main_gradients);

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
			std::set<std::string> &params,
			std::set<std::string> &computed_params,
			const std::string &prefix = "",
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
     * \brief creates a filler for each parameter defining the shape
     */
    void set_nclasses(const caffe2::NetDef &net, caffe2::NetDef &init, int nclasses);

    /**
     * \brief infers the number of classes based on the output's shape
     */
    int get_nclasses(const caffe2::NetDef &net, const caffe2::NetDef &init);

    /*
     *  Optimizers
     */

    // Optimizers are stored as functions that take a net
    // and use the gradients to update the parameters
    using Optimizer = std::function<
      void
      (const ModelContext&,	// context
       caffe2::NetDef&,		// net
       caffe2::NetDef&,		// init_net
       float,			// momentum
       float			// rms_decay
       )>;

    // List of registered optimizers : sgd, adagrad, adam, rmsprop
    // See caffe2/python/optimizer.py
    //XXX Add other optimizers: SparseAdagrad, RowWiseSparseAdagrad, etc.
    const Optimizer &get_optimizer(const std::string &name);

    /*
     *  Other net manipulations
     */

    /**
     * \brief truncates a net to keep only the operators and inputs needed to compute 'blob'
     *        This 'blob' is tagged as the external output of the net
     */
    void truncate_net(const caffe2::NetDef &net, caffe2::NetDef &out, const std::string &blob);

    // Same as above, except that the net is replaced by its truncated version
    void truncate_net(caffe2::NetDef &net, const std::string &blob);

    /**
     * \brief reads a .pb or .pbtxt file
     */
    void import_net(caffe2::NetDef &net, const std::string &file, bool unscoped = false);

    /**
     * \brief writes a .pb or .pbtxt file
     */
    void export_net(const caffe2::NetDef &net, const std::string &file, bool human_readable=false);

  }
}

#endif
