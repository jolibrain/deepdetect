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

#ifndef CPU_ONLY
#include <caffe2/core/context_gpu.h>
#endif

#include <caffe2/core/db.h>
#pragma GCC diagnostic pop

#include "caffe2/core/typeid.h"
#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Workspace initialization
     */

    ScopedNet ModelContext::scope_net(caffe2::NetDef &net) const {
      ScopedNet scoped(net);
      if (_parallelized) {
	scoped._devices = _devices;
      } else {
	scoped._devices = { _devices[0] };
	scoped._rename_inputs = scoped._rename_outputs = false;
      }
      return scoped;
    }

    void ModelContext::reset_devices() {
      _devices.clear();
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::DeviceTypeProto::PROTO_CPU);
      _devices.push_back(option);
      _parallelized = false;
    }

#ifndef CPU_ONLY
    void ModelContext::reset_devices(const std::vector<int> &gpu_ids) {
      _devices.clear();
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::DeviceTypeProto::PROTO_CUDA);
      for (int gpu_id : gpu_ids) {
	option.set_cuda_gpu_id(gpu_id);
	_devices.push_back(option);
      }
      _parallelized = (_devices.size() > 1);
    }
#endif

    bool ModelContext::extract_tensor(int device_idx,
				      const std::string &name,
				      caffe2::Tensor &tensor) const {
      const caffe2::Blob &blob = *_workspace->GetBlob(get_prefix(device_idx) + name);
      bool init = blob.meta().Match<caffe2::Tensor>();
      if (init) {
	tensor.CopyFrom(blob.Get<caffe2::Tensor>());
      }
      return init;
    }

    void ModelContext::insert_tensor(int device_idx,
				     const std::string &name,
				     const caffe2::Tensor &tensor) {
      caffe2::Blob &blob = *_workspace->CreateBlob(get_prefix(device_idx) + name);
      caffe2::DeviceType type(caffe2::ProtoToType(_devices[device_idx].device_type()));
      caffe2::BlobGetMutableTensor(&blob, type)->CopyFrom(tensor);
    }

    void ModelContext::broadcast_tensor(const std::string &name, const caffe2::Tensor &tensor) {
      for (size_t i = 0; i < device_count(); ++i) {
	insert_tensor(i, name, tensor);
      }
    }

    static bool deserialize_blob(caffe2::Blob &blob, const std::string &path) {
      std::ifstream f(path);
      CAFFE_ENFORCE(f.good());
      std::stringstream s;
      s << f.rdbuf();
      std::string str(s.str());
      if (str == caffe2::TypeMeta().name()) {
	return false; // nullptr (uninitialized)
      }
      caffe2::DeserializeBlob(str, &blob);
      return true;
    }

    void ModelContext::load_iter(const std::string &path) {
      caffe2::Blob blob;
      CAFFE_ENFORCE(deserialize_blob(blob, path));
      _loaded_iter = *blob.Get<caffe2::Tensor>().data<long>();
    }

    void ModelContext::load_blob(const std::string &path, const std::string &name) {
      CAFFE_ENFORCE(deserialize_blob(*_workspace->CreateBlob(name), path));
    }

    void ModelContext::load_lr(const std::string &path) {
      caffe2::Blob blob;
      // A learning rate serialized during the first iteration is still uninitialized
      // In that case we just ignore it
      if (deserialize_blob(blob, path)) {
	broadcast_tensor(blob_lr, blob.Get<caffe2::Tensor>());
      }
    }

    /*
     *  Information extraction
     */

    size_t ModelContext::extract_tensors(const std::string &name,
					 std::vector<caffe2::Tensor> &tensors) const {
      size_t size = 0;
      int devices = device_count();
      tensors.reserve(devices);
      for (int i = 0; i < devices; ++i) {
	tensors.emplace_back(caffe2::CPU);
	caffe2::Tensor &tensor(tensors[i]);
	CAFFE_ENFORCE(extract_tensor(i, name, tensor));
	size += tensor.size();
      }
      return size;
    }

    template <typename Result, typename Data>
    void ModelContext::split_tensors(std::vector<Result> &results,
				     const std::vector<caffe2::Tensor> &tensors,
				     const std::vector<size_t> &sizes,
				     const Stockage<Result, Data> &store) const {

      // Assign a value to each result
      typename std::vector<Result>::iterator result = results.begin();
      std::vector<size_t>::const_iterator size = sizes.begin();

      // Loop over the tensors
      for (const caffe2::Tensor &tensor : tensors) {
	CAFFE_ENFORCE(tensor.IsType<Data>());
	const Data *data = tensor.data<Data>();
	const Data *data_end = data + tensor.size();
	// Loop over the current batch
	for (; data < data_end; data += *size, ++result, ++size) {
	  store(*result, data, *size);
	}
      }
    }

    template <typename T>
    static void _store_single_value(T &result, const T *data, size_t size) {
      CAFFE_ENFORCE(size == 1);
      result = *data;
    }

    template <typename T>
    static void _store_vector(std::vector<T> &result, const T *data, size_t size) {
      result.assign(data, data + size);
    }

    template <typename Result, typename Data>
    static void assign_results(std::vector<Result> &results, const std::vector<Data> &datas) {
      results.assign(datas.begin(), datas.end());
    }

    template <typename Result, typename Data>
    static void assign_results(std::vector<std::vector<Result>> &results,
			       const std::vector<std::vector<Data>> &datas) {
      for (size_t i = 0; i < results.size(); ++i) {
	results[i].assign(datas[i].begin(), datas[i].end());
      }
    }

    template <typename T>
    void ModelContext::split_tensors(std::vector<T> &results,
				     const std::vector<caffe2::Tensor> &tensors,
				     const std::vector<size_t> &sizes) const {
      split_tensors(results, tensors, sizes, Stockage<T, T>(_store_single_value<T>));
    }

    template <typename T>
    void ModelContext::split_tensors(std::vector<std::vector<T>> &results,
				     const std::vector<caffe2::Tensor> &tensors,
				     const std::vector<size_t> &sizes) const {
      split_tensors(results, tensors, sizes, Stockage<std::vector<T>, T>(_store_vector<T>));
    }

    template <typename T>
    void ModelContext::extract_results(std::vector<T> &results,
				       const std::string &name,
				       size_t size) const {
      std::vector<caffe2::Tensor> tensors;
      size_t data_size = extract_tensors(name, tensors);
      size_t data_count = results.size();
      if (!size) {
	size = data_size / data_count;
      }
      CAFFE_ENFORCE(data_size == size * data_count);
      split_tensors(results, tensors, std::vector<size_t>(data_count, size));
    }

    template <typename T>
    void ModelContext::extract_results(std::vector<T> &results,
				       const std::string &name,
				       const std::vector<size_t> &sizes,
				       size_t scale) const {
      std::vector<caffe2::Tensor> tensors;
      size_t data_count = results.size();
      CAFFE_ENFORCE(data_count == sizes.size());
      size_t data_size1 = extract_tensors(name, tensors);
      size_t data_size2 = std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
      if (!scale && data_size1) {
	scale = data_size1 / data_size2;
      }
      CAFFE_ENFORCE(data_size1 == data_size2 * scale);
      std::vector<size_t> scaled_sizes(data_count);
      std::transform(sizes.begin(), sizes.end(), scaled_sizes.begin(),
		     [&](float size){ return scale * size; });
      split_tensors(results, tensors, scaled_sizes);
    }

    template <typename Result, typename Data, typename Size>
    void ModelContext::extract_and_cast_results(std::vector<Result> &results,
						const std::string &name,
						const Size &size) const {
      std::vector<Data> raw(results.size());
      extract_results(raw, name, size);
      assign_results(results, raw);
    }

    float ModelContext::extract_loss() const {
      std::vector<float> losses(device_count());
      extract_results(losses, blob_loss_scale, 1);
      return std::accumulate(losses.begin(), losses.end(), 0.f);
    }

    int ModelContext::extract_iter() const {
      std::vector<long> iters(device_count());
      extract_results(iters, blob_iter, 1);
      return iters[0];
    }

    void ModelContext::extract_labels(std::vector<float> &labels) const {
      extract_and_cast_results<float, int>(labels, _blob_label, 1);
    }

    void ModelContext::extract_state(std::map<std::string, std::string> &blobs) const {
      for (const std::string &name : _workspace->Blobs()) {
	const caffe2::Blob &blob = *_workspace->GetBlob(name);
    	if (blob.IsType<caffe2::db::DBReader>()) {
	  blobs[name] = caffe2::SerializeBlob(blob, name);
	}
      }
      const caffe2::Blob &iter(*_workspace->GetBlob(get_prefix(0) + blob_iter));
      blobs["iter"] = caffe2::SerializeBlob(iter, "iter");
      caffe2::Blob lr;
      if (extract_tensor(0, blob_lr, *caffe2::BlobGetMutableTensor(&lr, caffe2::CPU))) {
	blobs["lr"] = caffe2::SerializeBlob(lr, "lr");
      } else { // Can fail during the first iteration of the net
	blobs["lr"] = caffe2::TypeMeta().name(); // nullptr (uninitialized)
      }
    }

    void ModelContext::extract(std::vector<std::vector<float>> &results,
			       const std::string &name,
			       const std::vector<size_t> &sizes,
			       size_t scale) const {
      if (sizes.size()) {
	extract_results(results, name, sizes, scale);
      } else {
	extract_results(results, name);
      }
    }

    /*
     *  Network manipulation
     */

    void ModelContext::create_init_net(const caffe2::NetDef &net, caffe2::NetDef &init) const {
      std::set<std::string> params;
      caffe2::Tensor tensor(caffe2::CPU);
      collect_params(net, params, params, get_prefix(0));
      for (const std::string &param : params) {
	CAFFE_ENFORCE(extract_tensor(0, param, tensor));
	const float *data = tensor.data<float>();
	const std::vector<long int> &dims = tensor.dims();
	GivenTensorFill(init, param,
			std::vector<int>(dims.begin(), dims.end()),
			std::vector<float>(data, data + tensor.size()));
      }
    }

    void ModelContext::append_net(caffe2::NetDef &dst, const caffe2::NetDef &src) const {
      ScopedNet net = scope_net(dst);
      add_ops_and_inputs(net, src);
      dst.add_external_input(_input_blob);
    }

    void ModelContext::append_trainable_net(caffe2::NetDef &dst, const caffe2::NetDef &src,
					    const std::vector<std::string> &output_blobs) const {

      // Prevent the gradients from reaching previous operators
      ScopedNet net = scope_net(dst);
      StopGradient(net, _input_blob);

      // Assert that operators will generate the correct gradients
      caffe2::NetDef ref(src);
      for (caffe2::OperatorDef &op : *ref.mutable_op()) {
	for (caffe2::Argument &arg : *op.mutable_arg()) {
	  if (arg.name() == "is_test") {
	    arg.set_i(0);
	  }
	}
      }

      add_ops_and_inputs(net, ref, { _input_blob });

      //XXX Manage other kinds of outputs (no-label, multi-label, bbox, etc.)
      CAFFE_ENFORCE(output_blobs.size() == 1);
      LabelCrossEntropy(net, output_blobs[0], _blob_label, blob_xent);

      std::string loss_grad = blob_loss_scale + gradient_suffix;

      // Add the losses
      AveragedLoss(net, blob_xent, blob_loss);
      Scale(net, blob_loss, blob_loss_scale, 1.f / device_count());
      ConstantFill(net, blob_loss_scale, loss_grad, 1.0);

      // Add the gradients and sum them over the devices
      std::set<std::string> main_gradients;
      for (size_t i = 0; i < device_count(); ++i) {
	main_gradients.insert(get_prefix(i) + loss_grad);
      }
      add_gradient_ops(dst, main_gradients);
#ifndef CPU_ONLY
      if (device_count() > 1) {
	reduce(net);
      }
#endif
    }

    /*
     *  Other net manipulations
     */

    void truncate_net(const caffe2::NetDef &net, caffe2::NetDef &out, const std::string &blob) {
      auto ops = net.op();
      int last_op = find_previous_update(net, blob, ops.size() - 1);

      // Fill the new net
      for (int i = 0; i <= last_op; ++i) {
	out.add_op()->CopyFrom(ops[i]);
      }
      out.mutable_external_input()->CopyFrom(net.external_input());
      out.add_external_output(blob);
    }

    void truncate_net(caffe2::NetDef &net, const std::string &blob) {
      caffe2::NetDef short_net;
      truncate_net(net, short_net, blob);
      net.Swap(&short_net);
    }

    inline void unscope(std::string &s, const std::string &prefix) {
      if (!s.find(prefix)) {
	s.erase(0, prefix.size());
      }
    }

    void import_net(caffe2::NetDef &net, const std::string &file, bool unscoped) {
      CAFFE_ENFORCE(caffe2::ReadProtoFromFile(file, &net));
      if (unscoped) {
	// Some net templates begin with a prefix on each blob ('gpu_0/')
	// We may want to remove it early to prevent conflicts
	const char *prefix = "gpu_0/";
	for (std::string &s : *net.mutable_external_input()) unscope(s, prefix);
	for (std::string &s : *net.mutable_external_output()) unscope(s, prefix);
	for (caffe2::OperatorDef &op : *net.mutable_op()) {
	  for (std::string &s : *op.mutable_input()) unscope(s, prefix);
	  for (std::string &s : *op.mutable_output()) unscope(s, prefix);
	}
      }
    }

    void export_net(const caffe2::NetDef &net, const std::string &file, bool human_readable) {
      std::ofstream f(file);
      if (human_readable) {
	f << net.DebugString();
      } else {
	net.SerializeToOstream(&f);
      }
    }

    void append_model(caffe2::NetDef &dst_net, caffe2::NetDef &dst_init,
		      const caffe2::NetDef &src_net, const caffe2::NetDef &src_init) {
      for (auto op : src_init.op()) {
	dst_init.add_op()->CopyFrom(op);
	for (const std::string blob : op.output()) {
	  dst_net.add_external_input(blob);
	}
      }
      for (auto op : src_net.op()) {
	dst_net.add_op()->CopyFrom(op);
      }
    }

  }
}
