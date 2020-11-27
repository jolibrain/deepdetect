/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef TORCH_GRAPH_BACKEND_H
#define TORCH_GRAPH_BACKEND_H

#include "graph/basegraph.h"
#include <exception>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include <torch/ordered_dict.h>

namespace dd
{
  /**
   * \brief Generic exception for torch graph stuff
   */
  class TorchGraphException : public std::exception
  {
  public:
    TorchGraphException(const std::string &s) : _s(s)
    {
    }
    ~TorchGraphException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief torch module that includes a basegraph,
   * implements forward based on basegraph
   * redefines to() and parameters()
   * allocates torch modules based on embedded basegraph
   * should not be used as is
   * it only add torch::module traits ie to(), forward() parameters()...  to a
   * base graph
   */
  class TorchGraphBackend : public virtual graph::BaseGraph,
                            public torch::nn::Module
  {
  public:
    /**
     * \brief
     * empty constructor : do nothing this class is not meant to be constructed
     * alone
     *
     */
    TorchGraphBackend()
    {
    }

    /**
     * \brief torch::module forward :  inference of the NN
     * @param x input tensor
     * @return output tensor
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * \brief torch::module forward until some layer:  inference of the NN
     * @param x input tensor
     * @return output tensor
     */
    torch::Tensor extract(torch::Tensor x, std::string extract_layer);

    /**
     * \brief check is string correspond to some data in the net
     * @param the name of the data node
     * @return true if it exists in the net
     */
    bool extractable(std::string extract_layer);

    /**
     * \brief return all canidates for extraction, ie all data nodes of the net
     */
    std::vector<std::string> extractable_layers() const;

    /**
     * \brief allocates torch modules base on real input dimension
     * @param inputdim input tensor dimension, in int64_t
     */
    void finalize(std::vector<int64_t> inputdim);

    /**
     * \brief allocates torch modules base on real input dimension
     * @param inputdim input tensor dimension, in int
     */
    void finalize(std::vector<int> inputdim);

    /**
     * \brief allocates torch modules base on real input dimension
     * @param inputdim input tensor dimension, in torch :: IntArrayRef
     */
    void finalize(at::IntArrayRef inputdim);

    /**
     * \brief allocates torch modules base on current inputdim
     * hides basegraph::finalize() on purpose
     * generally should not be used except for very special cases
     */
    void finalize();

    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param non_blocking
     */
    virtual void to(torch::Device device, bool non_blocking = false);

    /**
     * \brief see torch::module::to
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    virtual void to(torch::Dtype dtype, bool non_blocking = false);

    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    virtual void to(torch::Device device, torch::Dtype dtype,
                    bool non_blocking = false);

    /**
     * see torch::modules::parameters()
     * override in order to warn if reallocation is needed while parameters are
     * used
     * @param recurse
     *
     * @return
     */
    std::vector<torch::Tensor> parameters(bool recurse = true);

    /**
     * set lstm continuation : if true then previous hidden state of lstm is
     * used for upcoming forward()
     * @param lc
     */
    void lstm_continues(bool lc)
    {
      _lstm_continuation = lc;
    }

    /**
     * informs torchgraphbackend that parameters are no more used,
     * ie reallocation can be done w/o warning
     */
    void parameters_release()
    {
      _parameters_used = false;
    }

    /**
     * tells if some allocation was done (needs to be called just after
     * set_inputdim or finalize
     */
    bool needs_reload()
    {
      return _allocation_done;
    }

    /**
     * return input dims from loaded tensor and not from
     * specification/construction for info/debug purposes
     */
    std::vector<int> get_input_dims_from_loaded();

  protected:
    /**
     * internal torch module allocation, called whithin (finalize)
     */
    void allocate_modules();

    std::unordered_map<std::string, torch::nn::AnyModule>
        _modules; /**< torch modules, per name/id */
    std::unordered_map<std::string, torch::Tensor>
        _variables; /**< torch intermediate tensors / blobs  */

    /**
     * \brief internal forward on basegraph
     * @param v vertex to forward
     * @return all outpurs
     */
    std::vector<torch::Tensor> forward(graph::Vertex v);

    /**
     * \brief internal set tensor input , used by forward
     * @param in   input tensor
     */
    void set_input(torch::Tensor in);

    std::vector<int>
    get_input_dims(std::string optype,
                   torch::OrderedDict<std::string, torch::Tensor> params);

    torch::Dtype _dtype
        = torch::kFloat32; /**< type of data stored in tensors */
    torch::Device _device
        = torch::DeviceType::CPU; /**< device to compute on */
    bool _parameters_used
        = false; /**< if parameters are use => warn if realloc */
    bool _lstm_continuation
        = false; /**< if lstm should use previous hidden state value */
    std::unordered_map<std::string, std::tuple<torch::Tensor, torch::Tensor>>
        _rnn_memories; /**< values of previsous hidden states */
    std::unordered_map<std::string, bool>
        _rnn_has_memories; /**< true if previsous hidden values are available
                            */

    bool _allocation_done = false;

    long int _autoencoder_timesteps = -1;
    /**< this var stores timesteps of an lstm autoencoder, read from lstm
     * layers and given to tile/repeat layer, in order not to have to put it in
     * prototxt */
  };
}

#endif
