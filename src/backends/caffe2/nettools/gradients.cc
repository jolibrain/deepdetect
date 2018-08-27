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
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <caffe2/core/operator.h>

#ifndef CPU_ONLY
#include <caffe2/core/context_gpu.h>
#endif

#pragma GCC diagnostic pop

#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Gradient management
     */

    bool is_trainable(const caffe2::OperatorDef &op,
		      std::set<std::string> *trainable = NULL,
		      std::set<std::string> *computed = NULL) {
      auto it = trainable_ops.find(op.type());
      if (it == trainable_ops.end()) {
	return false;
      }
      auto index = it->second.begin();
      auto end = it->second.end();
      int i = 0;
      for (const std::string &input : op.input()) {
	if (index == end || *index != i++) {
	  if (trainable) trainable->insert(input);
	} else {
	  ++index;
	  if (computed) computed->insert(input);
	}
      }
      return true;
    }

    /*
     *  When creating the gradients,
     *  the following informations are stored for the concerned blobs:
     *
     *		- TOTAL
     *			How many time this blob was used as an input.
     *			In other words, how many gradient will be used to compute
     *                  the 'backward-pass' version of this blob
     *
     *		- CURRENT
     *			Number of gradient that still need
     *			to be computed before this blob is complete
     *                  ( 0 <= CURRENT <= TOTAL )
     *
     *		- INPLACE
     *			How many of the previously mentionned gradient are sums that directly
     *                  update the blob with 'inplace' operation
     *
     *		- DEVICE
     *			GPU id on which the final gradient sum will be computed
     *			(set to -1 if there's no such device)
     *
     *  Below are some classes to make this more explicit in the code
     */

    class BlobInfo {
    public:

      int _current = 0;
      int _total = 0;
      int _inplace = 0;
      int _device = 0;

      inline void add(int inplace, int device) {
	++_current;
	++_total;
	_inplace += inplace;
	_device = device;
      }

      // When a blob isn't modified 'inplace', it is 'split' in several versions
      // (each one with a specific prefix and suffix)
      // When each 'split' is computed they're summed into the main blob.
      inline void rename_if_splitted(std::string &name, int i) {
	if (_total - _inplace > 1 && i) {
	  name = "_" + name + "_autosplit_" + std::to_string(i - 1);
	}
      }
    };

    class BlobsInfo: public std::map<std::string, BlobInfo> {
    public:

      inline bool is_tagged(const std::string &name) {
	return count(name);
      }

      // Fetch the next available 'split' of the given blob
      inline void rename_if_multiple_use(std::string &name) {

	if (!is_tagged(name)) return; // Can't cause conflicts

	BlobInfo &info = at(name);
	if (info._current < 1) return; // Won't be reused

	int split_nb = info._total - info._current;
	info._current--; // Use once
	info.rename_if_splitted(name, split_nb);
      }
    };

    // Because we want our gradients to be generated in the same way as caffe2's,
    // sometimes our gradient sums don't create a new output but are done 'inplace'.
    // To do this either inputs or output are renamed.
    // This is a map containing the renamed blobs.
    class RenamedBlobs: public std::map<std::string, std::string> {
    public:

      inline void rename_if_tagged(std::string &name) {
	auto it = find(name);
	if (it != end()) {
	  name = it->second;
	  erase(it);
	}
      }
    };

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static void collect_gradient_ops(caffe2::NetDef &net,
				     std::vector<caffe2::OperatorDef> &gradient_ops,
				     BlobsInfo &blobs_info) {

      std::set<std::string> external_inputs(net.external_input().begin(),
					    net.external_input().end());
      std::map<std::string, int> input_count;
      for (const caffe2::OperatorDef &op : net.op()) {

	if (!is_trainable(op)) {
	  // If we don't know whether an operator should or shouldn't be part of the gradient,
	  // we won't to the gradient at all and throw an error instead.
	  CAFFE_ENFORCE(non_trainable_ops.find(op.type()) != non_trainable_ops.end());
	  continue;
	}

	gradient_ops.push_back(op);
	int device = -1;
#ifndef CPU_ONLY
	if (op.device_option().device_type() == caffe2::CUDA) {
	  device = op.device_option().cuda_gpu_id();
	}
#endif
	for (const std::string &input : op.input()) {
	  const auto &output = op.output();
	  if (std::find(output.begin(), output.end(), input) != output.end()) {
	    // If a blob is tagged as both an input and an output,
	    // Then an inplace gradient will already be created (no need to it manually)
	    continue;
	  }
	  // If the blob is used by several operators, then its gradient will have multiple source.
	  // If those operators are Sums, then the gradient can be computed
	  // using inplace operations.
	  // If not we will need the manage a 'split' input and manually add a Sum.
	  blobs_info[input + gradient_suffix].add(op.type() == "Sum", device);
	}
      }
      // As we take an interest in the backward pass, we want the 'StopGradient' operator
      // to be put after the gradient-related blobs (and not before them).
      std::reverse(gradient_ops.begin(), gradient_ops.end());
    }

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static bool stop_op(const caffe2::OperatorDef &op, std::set<std::string> &stop_inputs) {
      bool stop = op.type() == "StopGradient";
      for (const std::string &output : op.output()) {
	stop |= stop_inputs.find(output) != stop_inputs.end();
      }
      if (stop) {
	for (const std::string &input : op.input()) {
	  stop_inputs.insert(input);
	}
      }
      return stop;
    }

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static void add_gradient_for_op(caffe2::NetDef &net, const caffe2::OperatorDef &op,
				    RenamedBlobs &renamed_blobs,
				    BlobsInfo &blobs_info) {
      // Feed the gradient with the operator outputs
      std::vector<caffe2::GradientWrapper> output(op.output_size());
      for (size_t i = 0; i < output.size(); ++i) {
	std::string grad = op.output(i) + gradient_suffix;
	if (blobs_info.is_tagged(grad)) { // This output must generate a gradient
	  output[i].dense_ = grad;
	}
	//XXX Manage sparse gradients (GradientWrapper.indices_ & GradientWrapper.values_)
	// See caffe2/python/core.py -- GradientRegistry._GetGradientForOpCC -- from_untyped
      }

      // Assert that gradients exist for this operator
      caffe2::GradientOpsMeta meta = GetGradientForOp(op, output);
      CAFFE_ENFORCE(meta.ops_.size());
      for (size_t i = 0; i < meta.ops_.size(); ++i) {
	caffe2::OperatorDef &grad = *net.add_op();
	grad.CopyFrom(meta.ops_[i]);
	grad.clear_name();
	grad.set_is_gradient_op(true);
	for (std::string &output : *grad.mutable_output()) {
	  blobs_info.rename_if_multiple_use(output);
	}
	for (std::string &input : *grad.mutable_input()) {
	  renamed_blobs.rename_if_tagged(input);
	}
      }
    }

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static void sum_gradients(caffe2::NetDef &net,
			      RenamedBlobs &renamed_blobs,
			      BlobsInfo &blobs_info) {
      for (auto &it : blobs_info) {
	const std::string &name = it.first;
	BlobInfo &info = it.second;
	if (info._current || info._total < 2) {
	  // current > 0:
	  //    Some operators still need to be computed before we do the final sum.
	  // current < 0:
	  //    The sum was already done.
	  // total < 2:
	  //    Nothing to sum
	  continue;
	}
	// Merge the 'split' versions of this blob
	std::vector<std::string> inputs;
	for (int i = 0; i < info._total; i++) {
	  std::string input = name;
	  info.rename_if_splitted(input, i);
	  renamed_blobs.rename_if_tagged(input);
	  if (input == name) {
	    inputs.insert(inputs.begin(), input); // Inplace
	  } else {
	    inputs.push_back(input);
	  }
	}
	caffe2::OperatorDef &op = Sum(net, inputs, name);
#ifndef CPU_ONLY
	if (info._device >= 0) {
	  op.mutable_device_option()->set_device_type(caffe2::CUDA);
	  op.mutable_device_option()->set_cuda_gpu_id(info._device);
	}
#endif
	// Setting counter to a negative value so it won't trigger anymore
	info._current--;
      }
    }

    void add_gradient_ops(caffe2::NetDef &net, const std::set<std::string> &main_gradients) {

      // Because we want our gradients to be generated in the same way as caffe2's,
      // sometimes our gradient sums don't create a new output but are done 'inplace'.
      // To do this either inputs or output are renamed.
      // This is a map containing the renamed blobs.
      RenamedBlobs renamed_blobs;

      // Ordered list of operators that can be needed to create the gradient.
      std::vector<caffe2::OperatorDef> gradient_ops;

      // Once the 'StopGradient' operator is reached,
      // this set is used to store uncomputable inputs.
      std::set<std::string> stop_inputs;

      // See the BlobInfo definition for more details
      BlobsInfo blobs_info;
      for (const std::string &grad : main_gradients) {
	blobs_info[grad]._total = 1;
      }

      collect_gradient_ops(net, gradient_ops, blobs_info);
      for (const caffe2::OperatorDef &op : gradient_ops) {
	// Operator placed after StopGradient must not be computed
	if (stop_op(op, stop_inputs)) {
	  continue;
	}

	if (op.type() == "Sum") {
	  // Make the first output as 'inplace' during the backward pass
	  for (const std::string &input : op.input()) {
	    std::string in = input + gradient_suffix;
	    blobs_info.rename_if_multiple_use(in);
	    renamed_blobs[in] = op.output(0) + gradient_suffix;
	  }
	} else {
	  add_gradient_for_op(net, op, renamed_blobs, blobs_info);
	}
	sum_gradients(net, renamed_blobs, blobs_info);
      }
    }

    /*
     *  Gradient & Device
     */

    void collect_params(const caffe2::NetDef &net,
			std::set<std::string> &params,
			std::set<std::string> &computed_params,
			const std::string &prefix,
			bool remove_prefix) {
      std::set<std::string> external_inputs(net.external_input().begin(),
					    net.external_input().end());
      for (const caffe2::OperatorDef &op : net.op()) {

	// Anything found before 'StopGradient' cannot be a parameter
	if (op.type() == "StopGradient") {
	  params.clear();
	  computed_params.clear();
	  continue;
	}

	std::set<std::string> trainable;
	std::set<std::string> computed;
	if (!is_trainable(op, &trainable, &computed)) {
	  continue;
	}

	auto check_params = [&](std::set<std::string> &s_in, std::set<std::string> &s_out) {
	  for (const std::string &input : s_in) {
	    if (!input.find(prefix) && external_inputs.find(input) != external_inputs.end()) {
	      s_out.insert(input.substr(remove_prefix * prefix.size(), -1));
	    }
	  }
	};
	check_params(trainable, params);
	check_params(computed, computed_params);
      }
    }

#ifndef CPU_ONLY

    // Check if gpus 'a' and 'b' have a quick access to each others
    static bool cuda_access_pattern(int a, int b) {
      static bool init = false;
      static std::vector<std::vector<bool> > access_pattern;
      if (!init) {
	CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&access_pattern));
	init = true;
      }
      return access_pattern[a][b];
    }

    /*
     *  Optimized tree reduction
     *  See pytorch/caffe2/python/data_parallel_model.py (_AllReduce function)
     */
    static void gradient_sum_order(const std::vector<int> &device_ids,
				   std::vector<std::vector<int> > &order) {
      int nb_devices = device_ids.size();
      if (nb_devices < 2 || nb_devices & (nb_devices - 1)) {
	order.push_back(device_ids);
	return;
      }

      for (int p = 1; p < nb_devices; p *= 2) {
	for (int i = 0; i < nb_devices; i += p * 2) {
	  order.push_back({device_ids[i], device_ids[i + p]});
	}
      }
    }

    void broadcast(ScopedNet &net, const std::string &blob) {

      std::vector<int> device_ids;
      for (const caffe2::DeviceOption &option : net._devices) {
	CAFFE_ENFORCE(option.device_type() == caffe2::CUDA);
	device_ids.push_back(option.cuda_gpu_id());
      }

      ScopeKeeper sk(net);
      net._rename_inputs = net._rename_outputs = false;
      if (blob == blob_iter) {
	caffe2::DeviceOption option;
	option.set_device_type(caffe2::CPU);
	net._devices = {option};
      }

      std::string main_blob = device_id_to_prefix(device_ids[0]) + blob;
      for (int device_id: device_ids) {
	if (device_id != device_ids[0]) {
	  net._force_device = device_id;
	  Copy(net, main_blob, device_id_to_prefix(device_id) + blob);
	}
      }
    }

    void reduce(ScopedNet &net) {

      std::vector<int> device_ids;
      std::set<std::string> params;
      std::set<std::string> computed_params;
      std::vector<std::vector<int> > sum_order;

      for (const caffe2::DeviceOption &option : net._devices) {
	CAFFE_ENFORCE(option.device_type() == caffe2::CUDA);
	device_ids.push_back(option.cuda_gpu_id());
      }

      // Create a backup
      // Both rename-related and device-related tags will be changed
      ScopeKeeper sk(net);
      net._rename_inputs = net._rename_outputs = false;

      gradient_sum_order(device_ids, sum_order);
      collect_params(net._net, params, computed_params, device_id_to_prefix(device_ids[0]), true);

      for (const std::string &param : params) {
	std::string gradient = param + gradient_suffix;
	for (const std::vector<int> &ordered_ids : sum_order) {

	  // The sum will be computed on the first gpu of the list
	  int host_id = ordered_ids[0];
	  net._force_device = host_id;
	  std::vector<std::string> blobs;

	  for (int other_id : ordered_ids) {

	    // Copy each versions of the gradient on the first gpu
	    // (except if cuda_access_pattern assert a quick communication over the gpus)
	    std::string blob = device_id_to_prefix(other_id) + gradient;
	    if (host_id == other_id || cuda_access_pattern(host_id, other_id)) {
	      blobs.push_back(blob);
	    } else {
	      std::string copy = device_id_to_prefix(host_id) + gradient +
		"_copy_" + std::to_string(other_id);
	      Copy(net, blob, copy);
	      blobs.push_back(copy);
	    }
	  }
	  Sum(net, blobs, blobs[0]);
	}
	broadcast(net, gradient);
      }
      for (const std::string &param : computed_params) {
	broadcast(net, param);
      }
    }
#endif

  }
}
