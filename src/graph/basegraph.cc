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

#include "operators.h"

namespace dd::graph
{
  Vertex BaseGraph::create_var(std::string name)
  {
    _finalized = false;
    _sorted = false;
    Vertex varVertex = boost::add_vertex(_graph);
    _graph[varVertex].name = name;
    _graph[varVertex].op = false;
    ;
    _varIndex[name] = varVertex;
    return varVertex;
  }

  Vertex BaseGraph::create_op(std::string name, std::string type)
  {
    _finalized = false;
    _sorted = false;
    Vertex l = boost::add_vertex(_graph);
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

  Vertex BaseGraph::set_input(std::string name, std::vector<int> &inputdim)
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
    Vertex v = add_layer(opname, optype);
    add_inputs(v, inputs);
    add_outputs(v, outputs);
  }

  Vertex BaseGraph::add_layer(std::string opname, std::string optype)
  {
    return create_op(opname, optype);
  }

  std::vector<Vertex> BaseGraph::add_outputs(std::string opname,
                                             std::vector<std::string> &outputs)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add output to non existent layer " << opname
                  << std::endl;
        std::vector<Vertex> ev;
        return ev;
      }
    return add_outputs(it->second, outputs);
  }

  std::vector<Vertex> BaseGraph::add_outputs(Vertex v,
                                             std::vector<std::string> &outputs)
  {
    std::vector<Vertex> outverts;
    for (std::string output : outputs)
      outverts.push_back(add_output(v, output));
    return outverts;
  }

  Vertex BaseGraph::add_output(std::string opname, std::string varname)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add output to non existent layer " << opname
                  << std::endl;
        return LONG_MAX;
      }
    return add_output(it->second, varname);
  }

  Vertex BaseGraph::add_output(Vertex v, std::string varname)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _varIndex.find(varname);
    Vertex varVertex;
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

  std::vector<Vertex> BaseGraph::add_inputs(std::string opname,
                                            std::vector<std::string> &inputs)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add inputs to non existent layer " << opname
                  << std::endl;
        std::vector<Vertex> ev;
        return ev;
      }
    return add_inputs(it->second, inputs);
  }

  std::vector<Vertex> BaseGraph::add_inputs(Vertex v,
                                            std::vector<std::string> &inputs)
  {
    std::vector<Vertex> iv;
    for (std::string input : inputs)
      iv.push_back(add_input(v, input));
    return iv;
  }

  Vertex BaseGraph::add_input(std::string opname, std::string varname)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _opIndex.find(opname);
    if (it == _opIndex.end())
      {
        std::cerr << "cannot add inputs to non existent layer " << opname
                  << std::endl;
        return LONG_MAX;
      }
    return add_input(it->second, varname);
  }

  Vertex BaseGraph::add_input(Vertex v, std::string varname)
  {
    std::unordered_map<std::string, Vertex>::iterator it
        = _varIndex.find(varname);
    Vertex varVertex;
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
    for (Vertex v : _sortedAll)
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

  void BaseGraph::compute_dims_operator(Vertex op)
  {
    std::vector<std::vector<int>> inputsdims;

    std::vector<Vertex> inputs;

    auto es = boost::in_edges(op, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      inputs.push_back(boost::source(*eit, _graph));

    for (unsigned int i = 0; i < inputs.size(); ++i)
      {
        std::vector<int> dims;
        for (int d : _graph[inputs[i]].dim)
          dims.push_back(d);
        inputsdims.push_back(dims);
      }

    std::vector<std::vector<int>> outputsdims;

    op::dispatcher::compute_outputs_dims(_graph[op], inputsdims, outputsdims);

    op::dispatcher::update_alloc_status(_graph[op], inputsdims, outputsdims);

    _graph[op].outputsdims = outputsdims;
    _graph[op].inputsdims = inputsdims;
  }

  void BaseGraph::compute_dims_var(Vertex var)
  {
    auto es = boost::in_edges(var, _graph);
    auto eit = es.first;
    Vertex producer = boost::source(*eit, _graph);

    auto ies = boost::out_edges(producer, _graph);
    unsigned int outputindex = 0;
    for (auto eit = ies.first; eit != ies.second; ++eit)
      {
        if (boost::target(*eit, _graph) == var)
          break;
        outputindex++;
      }

    _graph[var].dim = _graph[producer].outputsdims[outputindex];
  }

  void BaseGraph::compute_blob_sizes()
  {
    for (Vertex v : _sortedAll)
      {
        if (v == _input)
          continue;
        if (_graph[v].op)
          compute_dims_operator(v);
        else
          compute_dims_var(v);
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

  std::vector<Vertex> BaseGraph::inputs(Vertex v)
  {
    std::vector<Vertex> inputs;
    NNGraph::in_edge_iterator eit, eend;
    auto es = boost::in_edges(v, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      inputs.push_back(boost::source(*eit, _graph));
    return inputs;
  }

  std::vector<Vertex> BaseGraph::outputs(Vertex v)
  {
    std::vector<Vertex> outputs;
    NNGraph::out_edge_iterator eit, eend;
    auto es = boost::out_edges(v, _graph);
    for (auto eit = es.first; eit != es.second; ++eit)
      outputs.push_back(boost::target(*eit, _graph));
    return outputs;
  }

  int BaseGraph::dim(Vertex v, int num_input, int ndim)
  {
    return _graph[inputs(v)[num_input]].dim[ndim];
  }

  bool BaseGraph::allocated()
  {
    for (Vertex v : _sortedOps)
      if (_graph[v].alloc_needed)
        return false;
    return true;
  }
}
