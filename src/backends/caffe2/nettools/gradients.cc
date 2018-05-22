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
		      std::vector<std::string> *trainable = NULL,
		      std::vector<std::string> *computed = NULL) {
      auto it = trainable_ops.find(op.type());
      if (it == trainable_ops.end()) {
	return false;
      }
      auto index = it->second.begin();
      auto end = it->second.end();
      int i = 0;
      for (const std::string &input : op.input()) {
	if (index == end || *index != i++) {
	  if (trainable) trainable->push_back(input);
	} else {
	  ++index;
	  if (computed) computed->push_back(input);
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
     *  Below are some #define to make this more explicit in the code
     */

    using BlobInfo = std::tuple<int, int, int, int>; // <current, total, inplace, device>
#define BLOB_CURRENT(info) std::get<0>(info)
#define BLOB_TOTAL(info) std::get<1>(info)
#define BLOB_INPLACE(info) std::get<2>(info)
#define BLOB_DEVICE(info) std::get<3>(info)
#define SET_BLOB_INFO(info, value, inplace, device) {	\
      BLOB_CURRENT(info) = BLOB_TOTAL(info) = value;	\
      BLOB_INPLACE(info) += inplace;			\
      BLOB_DEVICE(info) = device;			\
    }

    // When a blob isn't modified 'inplace', it is 'split' in several versions
    // (each one with a specific prefix and suffix)
    // When each 'split' is computed they're summed into the main blob.
#define IS_SPLITTED(info) (BLOB_TOTAL(info) - BLOB_INPLACE(info) > 1)
#define SPLIT_NAME(name, i) if (i) {					\
      name = "_" + name + "_autosplit_" + std::to_string(i - 1);	\
    }

    // Fetch the next available 'split' of the given blob
#define RENAME_IF_MULTIPLE_USE(map, name)				\
    if (map.count(name)) {						\
      BlobInfo &info = map[name];					\
      if (BLOB_CURRENT(info) > 0) {					\
	int split_nb = BLOB_TOTAL(info) - BLOB_CURRENT(info);		\
	BLOB_CURRENT(info)--;						\
	if (IS_SPLITTED(info)) {					\
	  SPLIT_NAME(name, split_nb);					\
	}								\
      }									\
    }

    // See the 'pass_replace' map in the 'add_gradient_ops' function for an explanation.
#define RENAME_IF_TAGGED(map, name) {		\
      auto it = map.find(name);			\
      if (it != map.end()) {			\
	name = it->second;			\
	map.erase(it);				\
      }						\
    }

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static void collect_gradient_ops(caffe2::NetDef &net,
				     std::vector<caffe2::OperatorDef> &gradient_ops,
				     std::map<std::string, BlobInfo> &blobs_info) {

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
	  // If the blob is used by several operators,
	  // then its gradient will have multiple source (Sum)
	  if (++input_count[input] > 1) {
	    BlobInfo &info = blobs_info[input + gradient_suffix];
	    // If those operators are Sums, then the gradient can be computed
	    // using inplace operations.
	    // If they're not, we will need the manage a 'split' input
	    SET_BLOB_INFO(info, input_count[input], op.type() == "Sum", device);
	  }
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
				    std::map<std::string, std::string> &pass_replace,
				    std::map<std::string, BlobInfo> &blobs_info) {
      // Feed the gradient with the operator outputs
      std::vector<caffe2::GradientWrapper> output(op.output_size());
      for (size_t i = 0; i < output.size(); ++i) {
	output[i].dense_ = op.output(i) + gradient_suffix;
      }

      // Assert that a gradient exists for this operator
      caffe2::GradientOpsMeta meta = GetGradientForOp(op, output);
      CAFFE_ENFORCE(meta.ops_.size());
      for (size_t i = 0; i < meta.ops_.size(); ++i) {
	caffe2::OperatorDef &grad = *net.add_op();
	grad.CopyFrom(meta.ops_[i]);
	if (i) {
	  // If there are several operators,
	  // only the first will be linked to the other gradients
	  continue;
	}
	grad.set_is_gradient_op(true);
	for (std::string &output : *grad.mutable_output()) {
	  RENAME_IF_MULTIPLE_USE(blobs_info, output);
	}
	for (std::string &input : *grad.mutable_input()) {
	  RENAME_IF_TAGGED(pass_replace, input);
	}
      }
    }

    // Used once by add_gradient_ops
    // Moved in a function to make it more readable
    static void sum_gradients(caffe2::NetDef &net,
			      std::map<std::string, std::string> &pass_replace,
			      std::map<std::string, BlobInfo> &blobs_info) {
      for (auto &it : blobs_info) {
	const std::string &name = it.first;
	BlobInfo &info = it.second;
	if (BLOB_CURRENT(info)) {
	  // if > 0:
	  //    Some operators still need to be computed before we do the final sum.
	  // if < 0:
	  //    The sum was already done.
	  continue;
	}
	// Merge the 'split' versions of this blob
	std::vector<std::string> inputs;
	for (int i = 0; i < BLOB_TOTAL(info); i++) {
	  std::string input = name;
	  if (IS_SPLITTED(info)) {
	    SPLIT_NAME(input, i);
	  }
	  RENAME_IF_TAGGED(pass_replace, input);
	  if (input == name) {
	    inputs.insert(inputs.begin(), input); // Inplace
	  } else {
	    inputs.push_back(input);
	  }
	}
	caffe2::OperatorDef &op = Sum(net, inputs, name);
#ifndef CPU_ONLY
	if (BLOB_DEVICE(info) >= 0) {
	  op.mutable_device_option()->set_device_type(caffe2::CUDA);
	  op.mutable_device_option()->set_cuda_gpu_id(BLOB_DEVICE(info));
	}
#endif
	// Setting counter to a negative value so it won't trigger anymore
	--BLOB_CURRENT(info);
      }
    }

    void add_gradient_ops(caffe2::NetDef &net) {

      // Because we want our gradients to be generated in the same way as caffe2's,
      // sometimes our gradient sums don't create a new output but are done 'inplace'.
      // To do this either inputs or output are renamed.
      // This is a map containing the renamed blobs.
      std::map<std::string, std::string> pass_replace;

      // Ordered list of operators that can be needed to create the gradient.
      std::vector<caffe2::OperatorDef> gradient_ops;

      // Once the 'StopGradient' operator is reached,
      // this set is used to store uncomputable inputs.
      std::set<std::string> stop_inputs;

      // See the BlobInfo definition for more details
      std::map<std::string, BlobInfo> blobs_info;

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
	    RENAME_IF_MULTIPLE_USE(blobs_info, in);
	    pass_replace[in] = op.output(0) + gradient_suffix;
	  }
	} else {
	  add_gradient_for_op(net, op, pass_replace, blobs_info);
	}
	sum_gradients(net, pass_replace, blobs_info);
      }
    }

    /*
     *  Gradient & Device
     */

    void collect_params(const caffe2::NetDef &net,
			std::vector<std::string> &params,
			std::vector<std::string> &computed_params,
			const std::string &prefix,
			bool remove_prefix) {
      std::set<std::string> external_inputs(net.external_input().begin(),
					    net.external_input().end());
      for (const caffe2::OperatorDef &op : net.op()) {
	std::vector<std::string> trainable;
	std::vector<std::string> computed;
	if (!is_trainable(op, &trainable, &computed)) {
	  continue;
	}
	const auto &output = op.output();
#define CHECK_PARAMS(v_in, v_out)						\
	for (const std::string &input : v_in) {					\
	  if (!input.find(prefix) &&						\
	      external_inputs.find(input) != external_inputs.end() &&		\
	      std::find(output.begin(), output.end(), input) == output.end()) {	\
	    v_out.push_back(input.substr(remove_prefix * prefix.size(), -1));	\
	  }									\
	}
	CHECK_PARAMS(trainable, params);
	CHECK_PARAMS(computed, computed_params);
#undef CHECK_PARAMS
      }
    }

    void collect_params(const ScopedNet &net,
			std::vector<std::string> &params,
			std::vector<std::string> &computed_params,
			bool remove_prefix) {
      collect_params(net._net, params, computed_params,
		     get_device_prefix(net._devices[0]), remove_prefix);
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
      std::vector<std::string> params;
      std::vector<std::string> computed_params;
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
      collect_params(net, params, computed_params);

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
