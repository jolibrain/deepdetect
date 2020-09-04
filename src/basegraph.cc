/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "basegraph.h"
#include <iostream>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graphviz.hpp>
#include <iterator>
#include <climits>

namespace dd
{
  BaseGraph::Vertex BaseGraph::create_var(std::string name)
  {
    _finalized = false;
    _sorted = false;
    BaseGraph::Vertex varVertex = boost::add_vertex(_graph);
    _graph[varVertex].name = name;
    _graph[varVertex].op = false;
    ;
    _varIndex[name] = varVertex;
    return varVertex;
  }

  BaseGraph::Vertex BaseGraph::create_op(std::string name, std::string type)
  {
    _finalized = false;
    _sorted = false;
    BaseGraph::Vertex l = boost::add_vertex(_graph);
    _graph[l].name = name;
    _graph[l].type = type;
    _graph[l].op = true;
    _opIndex[name] = l;
    return l;
  }

  void BaseGraph::set_input_dim(std::vector<int> &inputdim)
  {
    if (_graph[_input].dim == inputdim)
      return;
    _graph[_input].dim = inputdim;
    _finalized = false;
  }

  BaseGraph::Vertex BaseGraph::set_input(std::string name,
                                         std::vector<int> &inputdim)
  {
    _inputname = name;
    _finalized = false;
    if (_varIndex.find(name) == _varIndex.end())
      _input = create_var(name);
    else
      _graph[_input].name = name;
    _graph[_input].dim = inputdim;
    return _input;
  }

  void BaseGraph::add_layer(std::string opname, std::string optype,
                            std::vector<std::string> &inputs,
                            std::vector<std::string> &outputs)
  {
    BaseGraph::Vertex v = add_layer(opname, optype);
    add_inputs(v, inputs);
    add_outputs(v, outputs);
  }

  BaseGraph::Vertex BaseGraph::add_layer(std::string opname,
                                         std::string optype)
  {
    return create_op(opname, optype);
  }

  std::vector<BaseGraph::Vertex>
  BaseGraph::add_outputs(std::string opname, std::vector<std::string> &outputs)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add output to non existent layer " << opname
                  << std::endl;
        std::vector<BaseGraph::Vertex> ev;
        return ev;
      }
    return add_outputs(it->second, outputs);
  }

  std::vector<BaseGraph::Vertex>
  BaseGraph::add_outputs(BaseGraph::Vertex v,
                         std::vector<std::string> &outputs)
  {
    std::vector<BaseGraph::Vertex> outverts;
    for (std::string output : outputs)
      outverts.push_back(add_output(v, output));
    return outverts;
  }

  BaseGraph::Vertex BaseGraph::add_output(std::string opname,
                                          std::string varname)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add output to non existent layer " << opname
                  << std::endl;
        return LONG_MAX;
      }
    return add_output(it->second, varname);
  }

  BaseGraph::Vertex BaseGraph::add_output(BaseGraph::Vertex v,
                                          std::string varname)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _varIndex.find(varname);
    BaseGraph::Vertex varVertex;
    if (it == _varIndex.end())
      varVertex = create_var(varname);
    else
      varVertex = it->second;
    if (boost::in_degree(varVertex, _graph) != 0)
      std::cerr << "trying to add multiple producers for  var "
                << _graph[varVertex].name << std::endl;
    else
      {
        boost::add_edge(v, varVertex, _graph);
        _finalized = false;
      }
    return varVertex;
  }

  std::vector<BaseGraph::Vertex>
  BaseGraph::add_inputs(std::string opname, std::vector<std::string> &inputs)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add inputs to non existent layer " << opname
                  << std::endl;
        std::vector<BaseGraph::Vertex> ev;
        return ev;
      }
    return add_inputs(it->second, inputs);
  }

  std::vector<BaseGraph::Vertex>
  BaseGraph::add_inputs(BaseGraph::Vertex v, std::vector<std::string> &inputs)
  {
    std::vector<BaseGraph::Vertex> iv;
    for (std::string input : inputs)
      iv.push_back(add_input(v, input));
    return iv;
  }

  BaseGraph::Vertex BaseGraph::add_input(std::string opname,
                                         std::string varname)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add inputs to non existent layer " << opname
                  << std::endl;
        return LONG_MAX;
      }
    return add_input(it->second, varname);
  }

  BaseGraph::Vertex BaseGraph::add_input(BaseGraph::Vertex v,
                                         std::string varname)
  {
    std::unordered_map<std::string, BaseGraph::Vertex>::iterator it
        = _varIndex.find(varname);
    BaseGraph::Vertex varVertex;
    if (it == _varIndex.end())
      varVertex = create_var(varname);
    else
      varVertex = it->second;
    _finalized = false;
    boost::add_edge(varVertex, v, _graph);
    return varVertex;
  }

  void BaseGraph::sort_all()
  {
    if (_sorted)
      return;
    _sortedOps.clear();
    _sortedVars.clear();
    boost::topological_sort(_graph, std::back_inserter(_sortedAll));
    std::reverse(_sortedAll.begin(), _sortedAll.end());
    for (BaseGraph::Vertex v : _sortedAll)
      {
        if (_graph[v].op)
          _sortedOps.push_back(v);
        else
          _sortedVars.push_back(v);
      }
    _sorted = true;
  }

  void BaseGraph::finalize()
  {
    if (_finalized)
      return;
    // TODO :  some sanity checks,
    // is there a path from input to ouput ?
    // check if all necessary vars are produced
    //
    sort_all();
    compute_blob_sizes();
    _finalized = true;
  }

  std::tuple<std::vector<int>, std::vector<int>>
  BaseGraph::compute_dims_from_producer(BaseGraph::Vertex producer)
  {
    std::vector<int> outputdim;
    std::vector<int> inputdim;
    std::vector<BaseGraph::Vertex> inputs;
    auto es = boost::in_edges(producer, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      inputs.push_back(boost::source(*eit, _graph));
    outputdim.push_back(_graph[inputs[0]].dim[0]); // batchsize
    for (int d : _graph[inputs[0]].dim)
      inputdim.push_back(d);
    if (_graph[producer].type == "LSTM" || _graph[producer].type == "RNN")
      {
        outputdim.push_back(_graph[inputs[0]].dim[1]);    // timesteps
        outputdim.push_back(_graph[producer].num_output); // hidden_size
      }
    else if (_graph[producer].type == "InnerProduct")
      {
        unsigned int axis
            = (_graph[producer].axis == -1) ? 1 : _graph[producer].axis;
        for (unsigned int i = 1; i < axis; ++i)
          outputdim.push_back(_graph[inputs[0]].dim[i]);
        outputdim.push_back(_graph[producer].num_output);
      }
    return std::tie(inputdim, outputdim);
  }

  void BaseGraph::compute_dims(BaseGraph::Vertex v)
  {
    auto es = boost::in_edges(v, _graph);
    auto eit = es.first;
    BaseGraph::Vertex producer = boost::source(*eit, _graph);
    auto newdims = compute_dims_from_producer(producer);
    update_alloc_status(newdims, producer);
    _graph[producer].dim = std::get<1>(newdims);
    _graph[v].dim = std::get<1>(newdims);
    _graph[producer].inputdim = std::get<0>(newdims);
  }

  void BaseGraph::update_alloc_status(
      std::tuple<std::vector<int>, std::vector<int>> newdims,
      BaseGraph::Vertex v)
  {
    std::vector<int> old_outputdims = _graph[v].dim;
    std::vector<int> old_inputdims = _graph[v].inputdim;
    std::vector<int> new_outputdims = std::get<1>(newdims);
    std::vector<int> new_inputdims = std::get<0>(newdims);
    if (old_outputdims.size() != new_outputdims.size())
      {
        _graph[v].alloc_needed = true;
        return;
      }

    if (old_inputdims.size() != new_inputdims.size())
      {
        _graph[v].alloc_needed = true;
        return;
      }

    // for lstm, rnn layers, special case for timesteps => no realloc needed
    if ((old_inputdims[1] != new_inputdims[1]))
      {
        if (_graph[v].type != "LSTM" && _graph[v].type != "RNN"
            && (_graph[v].type != "InnerProduct" || _graph[v].axis <= 1))
          {
            _graph[v].alloc_needed = true;
            return;
          }
      }

    // for every layer, if input dim > 2 , realloc needed
    for (unsigned int i = 2; i < new_inputdims.size(); ++i)
      if (old_inputdims[i] != new_inputdims[i])
        {
          _graph[v].alloc_needed = true;
          return;
        }

    if ((old_outputdims[1] != new_outputdims[1]))
      {
        if (_graph[v].type != "LSTM" && _graph[v].type != "RNN"
            && (_graph[v].type != "InnerProduct" || _graph[v].axis <= 1))
          {
            _graph[v].alloc_needed = true;
            return;
          }
      }
    for (unsigned int i = 2; i < old_outputdims.size(); ++i)
      if (old_outputdims[i] != new_outputdims[i])
        {
          _graph[v].alloc_needed = true;
          return;
        }
  }

  void BaseGraph::compute_blob_sizes()
  {
    for (BaseGraph::Vertex v : _sortedVars)
      {
        if (v == _input)
          continue;
        compute_dims(v);
      }
  }

  void BaseGraph::todot(std::ostream &out)
  {
    boost::write_graphviz(
        out, _graph,
        make_vertex_writer(boost::get(&VertexProperty::name, _graph),
                           boost::get(&VertexProperty::op, _graph),
                           boost::get(&VertexProperty::dim, _graph)));
  }

  std::vector<BaseGraph::Vertex> BaseGraph::inputs(BaseGraph::Vertex v)
  {
    std::vector<BaseGraph::Vertex> inputs;
    NNGraph::in_edge_iterator eit, eend;
    auto es = boost::in_edges(v, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      inputs.push_back(boost::source(*eit, _graph));
    return inputs;
  }

  std::vector<BaseGraph::Vertex> BaseGraph::outputs(BaseGraph::Vertex v)
  {
    std::vector<BaseGraph::Vertex> outputs;
    NNGraph::out_edge_iterator eit, eend;
    auto es = boost::out_edges(v, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      outputs.push_back(boost::target(*eit, _graph));
    return outputs;
  }

  int BaseGraph::dim(BaseGraph::Vertex v, int num_input, int ndim)
  {
    return _graph[inputs(v)[num_input]].dim[ndim];
  }

  bool BaseGraph::allocated()
  {
    for (BaseGraph::Vertex v : _sortedOps)
      if (_graph[v].alloc_needed)
        return false;
    return true;
  }
}
