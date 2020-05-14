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

#ifndef BASE_GRAPH_H
#define BASE_GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <ostream>

namespace dd
{

  /**
   * base graph is the high level representation of the neural net and soon other ops (chains/ data aug)
   *
   */
  class BaseGraph
  {

  public:

	/**
	 *  a vertex contains info about the operator on the computation flow
	 *  a vertex is also used for data blobs, in that case op is set to false
	 */
	struct VertexProperty
	{
	  bool op; /**< false if this vertex is data only (used for split / dup nodes) */
	  std::string name;	 /**< name of the operator / data tensor */
	  std::string type; /**< type of the operator  */
	  std::vector<int> dim;  /**< output dim for operators, data dim for data blobs */
	  std::vector<int> inputdim; /**< input dims for operators, needed to see if some change has occured */
	  bool alloc_needed = true; /**< true if some dimensions have changed */
	  int num_output = -1;		/**< number of output for some operator types */
	  int axis = -1;			/**< axis parameters for innerproduct operator type */
	};

	/**
	 * \brief edge property holder : nothing atm
	 *
	 */
	struct EdgeProperty
	{
	};

	/**
	 * \brief highly templated functor for having a string representation of a vertex
	 *
	 */
	template <class NameMap, class OpMap, class DimMap>
	class vertex_writer {
	public:
	  vertex_writer(NameMap n, OpMap t, DimMap d) : name(n), op(t), dim(d) {}
	  template <class Vertex>
	  void operator()(std::ostream &out, const Vertex& v) const {
		if (op[v])
		  out << "[label=\"" << name[v] << "\", shape=box]";
		else
		  {
			std::string ds = "{ ";
			std::vector<int> dims = dim[v];
			ds += std::to_string(dims[0]);
			for (unsigned int i = 1; i<dims.size(); ++i)
			  ds += " , " + std::to_string(dims[i]);
			ds+= " }";
			out << "[label=\"" << name[v] << " " << ds << "\", shape=ellipse]";
		  }
	  }
	private:
	  NameMap name;
	  OpMap op;
	  DimMap dim;
	};

	/**
	 * factory functor for debug, or something like this
	 */
	template <class NameMap, class OpMap, class DimMap>
	inline vertex_writer<NameMap, OpMap, DimMap> make_vertex_writer(NameMap n, OpMap t, DimMap d) {
	  return vertex_writer<NameMap, OpMap, DimMap>(n,t,d);
	}



	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperty, EdgeProperty> NNGraph; /**< the graph itself */
	typedef boost::graph_traits<NNGraph>::vertex_descriptor Vertex; /**< append vertex info */
	typedef boost::graph_traits<NNGraph>::edge_descriptor Edge; /**< append edge info (nothing atm) */


	/**
	 * \brief add layer / operator to the computational graph
	 *
	 * @param opname name of the layer
	 * @param optype type of the layer
	 * @param inputs name of inputs to connect
	 * @param outputs name of outputs
	 */
	void add_layer(std::string opname, std::string optype,
				   std::vector<std::string>& inputs,
				   std::vector<std::string>& outputs);

	/**
	 * \brief add layer / operator to the computational graph, reduced version
	 * @param opname name of the layer
	 * @param optype type of the layer
	 * @return  added vertex
	 */
	Vertex add_layer(std::string opname, std::string optype);

	/**
	 * \brief set general input
	 * @param name name of input data/blob
	 * @param inputdim dimensions of the blob
	 * @return the added vertex
	 */
	Vertex set_input(std::string name, std::vector<int>& inputdim);

	/**
	 * \brief add/bind some inputs to a vertex
	 * if they already exist, they are bound, if not they are allocated
	 * @param v vertex to add inputs to
	 * @param inputs name of inputs
	 * @return list of data inputs
	 */
	std::vector<Vertex> add_inputs(Vertex v, std::vector<std::string>& inputs);

	/**
	 * \brief add/bind some inputs to a vertex
	 * if they already exist, they are bound, if not they are allocated
	 * @param opname name of operator to add inputs to
	 * @param inputs name of inputs
	 * @return list of data inputs
	 */
	std::vector<Vertex> add_inputs(std::string opname, std::vector<std::string>& inputs);

	/**
	 * \brief add single input to a vertex, allocate data vertex if needed
	 * @param v vertex to add input to
	 * @param input name of data input
	 * @return input vertex
	 */
	Vertex add_input(Vertex v, std::string input);


	/**
	 * \brief add single input to a vertex, allocate data vertex if needed
	 * @param opname operator to add input to
	 * @param input name of data input
	 * @return input vertex
	 */
	Vertex add_input(std::string opname, std::string inputs);

	/**
	 * \brief add outputs to an operator vertex, allocates outputs if needed
	 * @param v vertex to add ouput to
	 * @param outputs name of data outputs
	 * @return vector of data vertices
	 */
	std::vector<Vertex> add_outputs(Vertex v, std::vector<std::string>& outputs);


	/**
	 * \brief add outputs to an operator , allocates outputs if needed
	 * @param opname name of operator to add ouput to
	 * @param outputs name of data outputs
	 * @return vector of data vertices
	 */
	std::vector<Vertex> add_outputs(std::string opname, std::vector<std::string>& outputs);


	/**
	 * \brief add output to an operator vertex, allocates output if needed
	 * @param v vertex to add ouput to
	 * @param varname name of data output
	 * @return data vertex
	 */
	Vertex add_output(Vertex v, std::string varname);

	/**
	 * \brief add output to an operator , allocates output if needed
	 * @param opnameto add ouput to
	 * @param varname name of data output
	 * @return data vertex
	 */
	Vertex add_output(std::string opname, std::string varname);

	/**
	 * \brief set loss to be used at learning
	 * @param name  loss name
	 */
	void set_loss(std::string name) {_loss= name;}

	/**
	 * topological_sort of vertices, used internally
	 */
	void sort_all();

	/**
	 * \brief gives a dot reprensention of the graph, for debgging purposes
	 * @param out ostream to put text to
	 */
	void todot(std::ostream &out);

	/**
	 * \brief check, sort, do some computations on blob sizes
	 */
	void finalize();

	/**
	 * \brief set input data dimensions
	 * @param inputdim
	 */
	void set_input_dim(std::vector<int>& inputdim);


	/**
	 * \brief set ouput  data name
	 * @param output : data blob name
	 */
	void set_output_name(std::string output) {_outputname = output;}

	/**
	 * \brief get input vertices
	 * @param Vertex from which inputs are needed
	 * @return all inpit vertices
	 */
	std::vector<Vertex> inputs(Vertex);

	/**
	 * \brief get output vertices
	 * @param Vertex from which outputs are needed
	 * @return all output vertices
	 */
	std::vector<Vertex> outputs(Vertex);

	/**
	 * \brief get some precise dimension of some inpit of a vertex
	 *
	 * @param v vertex from which some input dimension is needed
	 * @param num_input number of the input (generally 0 for first input)
	 * @param ndim number of dimension needed
	 *
	 * @return the dimension
	 */
	int dim(Vertex v, int num_input, int ndim);

	/**
	 * \brief gives number of outputs of a vertex
	 */
	inline
	int num_output(Vertex v) {return _graph[v].num_output;}

	/**
	 * \brief accessor for an operator vertex name
	 */
	inline
	std::string opname(Vertex v) {return _graph[v].name;}

	/**
	 * \brief accessor for a varaible vertex name
	 */
	inline
	std::string varname(Vertex v) {return _graph[v].name;}

	/**
	 * \brief accessor for an operator vertex type
	 */
	inline
	std::string optype(Vertex v) {return _graph[v].type;}


  protected:
	NNGraph _graph;				/**< the graph is here !  */
	Vertex _input;				/**<  root input vertex */
	std::string _inputname; /**<  root input name */
	std::string _outputname; /**<  ouput name */
	std::string _loss;			/**< loss name */

	std::map<std::string, Vertex> _varIndex; /**< to get a data vertex from its name */
	std::map<std::string, Vertex> _opIndex; /**< to get an operator vertex from its name */

	/**
	 * \brief helper to create a data vertex
	 * @param name
	 */
	Vertex create_var(std::string name);

	/**
	 * hleper to create an operator vertex
	 * @param name
	 * @param type
	 * @return
	 */
	Vertex create_op(std::string name, std::string type);

	std::vector<Vertex> _sortedAll; /**< all vertices, sorted  */
	std::vector<Vertex> _sortedOps; /**< all operators, sorted */
	std::vector<Vertex> _sortedVars; /**< all data blobs, sorted */

	bool _finalized =false;		/**< is laready finalized?  */
	bool _sorted = false;		/**< is already sorted?  */

	/**
	 * \brief compute input/output dims of a data vertex, given is producer,
	 * @param producer
	 * @param var
	 * @return input dim of producer, output dims of producer == dims of var
	 */
	std::tuple<std::vector<int>,std::vector<int>> compute_dims(Vertex producer, Vertex var);

	/**
	 * \brief compute dims of a data vertex
	 * @param v
	 */
	void compute_dims(Vertex v);

	/**
	 * compute all data vertices sizes
	 */
	void compute_blob_sizes();

	/**
	 * \brief check if realloc is needed, in case of change of input/ouput dims
	 */
	void update_alloc_status(std::tuple<std::vector<int>, std::vector<int>> newdims,
							 Vertex v);

	/**
	 * check if everything is correclty allocated, ie do no need reallocation
	 * @return
	 */
	bool allocated();
  };
}



#endif
