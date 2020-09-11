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

namespace dd
{
  namespace Caffe2NetTools
  {

    /*
     *  Parameter fillers
     */

    // Fetch the 'idx'th output of an operator
    // Create the requested filler for it, and place it in the 'fillers' map
    static caffe2::OperatorDef *
    add_filler(const caffe2::OperatorDef &op,
               std::map<std::string, caffe2::OperatorDef> &fillers,
               const std::string &ftype, int input_idx)
    {
      caffe2::OperatorDef *filler = NULL;
      if (input_idx < op.input().size())
        {
          const std::string &output = op.input(input_idx);
          filler = &fillers[output];
          filler->set_type(ftype);
          filler->add_output(output);
        }
      return filler;
    }

    // Same as above, but also adds a constant value to the filler
    static caffe2::OperatorDef *
    add_filler(const caffe2::OperatorDef &op,
               std::map<std::string, caffe2::OperatorDef> &fillers,
               const std::string &ftype, int input_idx, float value)
    {
      caffe2::OperatorDef *filler = add_filler(op, fillers, ftype, input_idx);
      if (filler)
        {
          add_arg(*filler, "value", value);
        }
      return filler;
    }

    // Filler functions take an operator, and create a filler for each input
    // referenced as being a parameter of the net. Thoses inputs and their
    // newly created fillers are storeds in the 'fillers' map.
    using Filler = std::function<void(
        const caffe2::OperatorDef &,                 // Input
        std::map<std::string, caffe2::OperatorDef> & // Output
        )>;

    // Just to make it more readable
#define FILLER_LAMBDA                                                         \
  [](const caffe2::OperatorDef &o,                                            \
     std::map<std::string, caffe2::OperatorDef> &f)
#define ADD_FILLER(type, args...) add_filler(o, f, #type "Fill", args)

    // Map linking operator types to their filler lambda
    static const std::map<std::string, Filler> op_fillers ({
	{
	  "Conv", FILLER_LAMBDA {
	    ADD_FILLER(Xavier, 1);
	    ADD_FILLER(Constant, 2);
  }
}
, { "FC", FILLER_LAMBDA{ ADD_FILLER(Xavier, 1);
ADD_FILLER(Constant, 2);
}
}
,
{
  "SpatialBN", FILLER_LAMBDA
  {
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
static void
collect_filler_types(const caffe2::NetDef &net,
                     std::map<std::string, caffe2::OperatorDef> &fillers)
{
  for (const caffe2::OperatorDef &op : net.op())
    {
      auto it = op_fillers.find(op.type());
      if (it != op_fillers.end())
        {
          it->second(op, fillers);
        }
    }
}

// Replace the net current fillers (usually GivenTensorFills) with the ones
// referenced in the 'fillers' map
static void
apply_filler_types(caffe2::NetDef &net,
                   const std::map<std::string, caffe2::OperatorDef> &fillers)
{
  for (caffe2::OperatorDef &op : *net.mutable_op())
    {
      auto it = fillers.find(op.output(0));
      if (it == fillers.end())
        {
          continue;
        }
      caffe2::OperatorDef copy = it->second;
      for (const caffe2::Argument &arg : op.arg())
        {
          if (arg.name() == "shape")
            {
              copy.add_arg()->CopyFrom(arg);
              break;
            }
        }
      op.Swap(&copy);
    }
}

/*
 * \brief browse the net and list shapes of blobs initialized by filling
 * operators
 * @param net net to browse
 * @param shapes used to store blobs name and shape
 * @param prefix filter out blob names that don't start with this prefix
 * @param remove_prefix whether the given prefix must be removed from the blob
 * name
 */
static void
collect_blob_shapes(const caffe2::NetDef &net,
                    std::map<std::string, std::vector<int>> &shapes,
                    const std::string &prefix = "", bool remove_prefix = true)
{
  for (const caffe2::OperatorDef &op : net.op())
    {
      auto it = non_trainable_ops.find(op.type());
      if (it == non_trainable_ops.end() || !it->second
          || // Not a filling operator
          op.input().size() || op.output().size() != 1
          || // Shape not stored in arguments
          op.output(0).find(prefix))
        { // Does not start with the given prefix
          continue;
        }
      std::string output
          = op.output(0).substr(remove_prefix * prefix.size(), -1);
      for (const caffe2::Argument &arg : op.arg())
        {
          if (arg.name() == "shape")
            {
              shapes[output].assign(arg.ints().begin(), arg.ints().end());
              break;
            }
        }
    }
}

// Some values (e.g. the number of classes) are present several times inside
// init nets. This class stores pointers to integers that represent the same
// value. Some of them may be multiples of one another, so a coefficient is
// linked to each pointer.
class ShapePtrs
{
  std::vector<std::pair<const int *, float>> _ptrs;

  // For the concerned blobs, two informations are stored:
  //   - Which dimension of its shape corresponds to the desired value
  //   - Its coefficient
  std::map<std::string, std::pair<int, float>> _blobs;

  // Store a pointer to the nth dimension of an input filler
  void add_ptr(const caffe2::OperatorDef &filler, int dimension, float coef)
  {

    // Find the shape
    for (const caffe2::Argument &arg : filler.arg())
      {
        if (arg.name() != "shape")
          {
            continue;
          }

        // Shapes should be small enough to fit into an integer (stored as long
        // in caffe2)
        if (dimension < arg.ints().size())
          {
            return _ptrs.emplace_back(
                reinterpret_cast<const int *>(&arg.ints()[dimension]), coef);
          }

        CAFFE_THROW("Can't access ", filler.output(0), " shape at position ",
                    dimension);
      }
    CAFFE_THROW("Can't access ", filler.output(0), " shape");
  }

protected:
  inline void register_blob(const std::string &blob, int dimension,
                            float coefficient)
  {
    _blobs[blob] = { dimension, coefficient };
  }

  void fetch_pointers(const caffe2::NetDef &init_net)
  {

    // Make a copy
    std::map<std::string, std::pair<int, float>> blobs = _blobs;

    // Check if the fillers output are registered
    for (const caffe2::OperatorDef &filler : init_net.op())
      {
        const auto &blob = blobs.find(filler.output(0));
        if (blob == blobs.end())
          {
            continue;
          }

        // Keep a pointer to the shape
        add_ptr(filler, blob->second.first, blob->second.second);
        blobs.erase(blob);
      }

    // Assert that every blob was found
    const auto &blob = blobs.begin();
    if (blob != blobs.end())
      {
        CAFFE_THROW("Can't access ", blob->first, " shape");
      }

    // Assert that the retreived data is coherent
    int value = *_ptrs[0].first / _ptrs[0].second;
    for (const std::pair<const int *, int> &ptr : _ptrs)
      {
        if (*ptr.first % ptr.second || *ptr.first / ptr.second != value)
          {
            CAFFE_THROW("Couldn't fetch data from the initalization net");
          }
      }
  }

public:
  inline void get_blobs(std::set<std::string> &blobs)
  {
    blobs.clear();
    for (const std::pair<std::string, std::pair<int, float>> &it : _blobs)
      {
        blobs.insert(it.first);
      }
  }

  inline int get_value() const
  {
    return *_ptrs[0].first / _ptrs[0].second;
  }

  void set_value(int value)
  {
    if (get_value() != value)
      {
        for (const std::pair<const int *, int> ptr : _ptrs)
          {
            *const_cast<int *>(ptr.first) = value * ptr.second;
          }
      }
  }
};

// Group of function used to find which integers are defining the output shape
// of a net Also contains maps used to know how an operator output shape is
// defined (detailed right after the class)
class OutputShapePtrs : public ShapePtrs
{

  /*
   *  Some operators have their shape defined by previous operators.
   */
  //                    Operator              Input, Coef
  static const std::map<std::string, std::map<int, float>> _forwarded_shapes;

  /*
   *  Other operators have their shape defined by external inputs.
   *  Thoses shapes too may be multiples of one another, but it will be
   * explicitly shown inside the 'shape' argument of the corresponding filler.
   * Unfortunately we still need to know which dimension of thoses 'shape's is
   * shared.
   */
  //                    Operator              Input, Dimension index
  static const std::map<std::string, std::map<int, int>> _external_shapes;

  // Recursive function that start browsing the net a specific index and
  // coefficient
  void compute(int op_idx, float coef)
  {
    const caffe2::OperatorDef &op = _net.op(op_idx);
    const std::string &type = op.type();

    if (_forwarded_shapes.find(type) != _forwarded_shapes.end())
      {
        // Shape defined by previous operators

        for (const std::pair<int, float> &input_coef :
             _forwarded_shapes.at(type))
          {
            // Recurse over them
            const std::string &input_name
                = _net.op(op_idx).input(input_coef.first);
            compute(find_previous_update(_net, input_name, op_idx - 1),
                    coef * input_coef.second);
          }
      }
    else if (_external_shapes.find(type) != _external_shapes.end())
      {
        // Shape defined by external inputs

        for (const std::pair<int, int> &input_dim : _external_shapes.at(type))
          {
            int input_idx = input_dim.first;
            if (input_idx >= op.input().size())
              {
                CAFFE_THROW("Can't access ", op.type(), " input at position ",
                            input_idx);
              }
            // Save the blob informations
            register_blob(op.input(input_idx), input_dim.second, coef);
          }
      }
    else
      {
        // XXX Handle more configurations
        CAFFE_THROW("Can't access ", op.type(), " input shapes");
      }
  }

public:
  const caffe2::NetDef &_net;
  OutputShapePtrs(const caffe2::NetDef &net, const caffe2::NetDef &init_net)
      : _net(net)
  {
    // Start computing from the last operator whose shape is (by definition)
    // the same as the net output shape (so coef=1)
    compute(_net.op().size() - 1, 1);
    fetch_pointers(init_net);
  }
};

const std::map<std::string, std::map<int, float>>
    OutputShapePtrs::_forwarded_shapes({

        // One input, same shape
        { "AveragePool",
          { { 0, 1 } } }, // (An end-of-net pooling should be a global pooling)
        { "CopyFromCPUInput", { { 0, 1 } } },
        { "EnsureCPUOutput", { { 0, 1 } } },
        { "Relu", { { 0, 1 } } },
        { "Sigmoid", { { 0, 1 } } },
        { "Softmax", { { 0, 1 } } },

        // An input that is already a group of bbox
        { "BBoxTransform", { { 1, 1 } } },

        // 1 probablity per result
        // 4 points per result
        { "BoxWithNMSLimit", { { 0, 1 }, { 1, 4 } } }

    });

const std::map<std::string, std::map<int, int>>
    OutputShapePtrs::_external_shapes({

        // Weights have 2 dimensions (nb_output and nb_input), we want the
        // first
        // Bias have 1 dimension (nb_output), we also want the first
        { "Conv", { { 1, 0 }, { 2, 0 } } },
        { "FC", { { 1, 0 }, { 2, 0 } } } });

// XXX Used by debug.cc
void _reset_init_net(const caffe2::NetDef &net, caffe2::NetDef &init)
{
  std::map<std::string, caffe2::OperatorDef> fillers;
  collect_filler_types(net, fillers);
  apply_filler_types(init, fillers);
}

void set_nclasses(const caffe2::NetDef &net, caffe2::NetDef &init,
                  int nclasses)
{

  // Reshape the blobs defining the output shape
  std::set<std::string> shape;
  OutputShapePtrs ptrs(net, init);
  ptrs.set_value(nclasses);
  ptrs.get_blobs(shape);

  // Collect every fillers
  std::map<std::string, caffe2::OperatorDef> fillers;
  collect_filler_types(net, fillers);

  // Filter them
  for (auto it = fillers.begin(); it != fillers.end();)
    {
      if (shape.find(it->first) == shape.end())
        {
          it = fillers.erase(it);
        }
      else
        {
          ++it;
        }
    }

  // Reset them
  apply_filler_types(init, fillers);
}

int get_nclasses(const caffe2::NetDef &net, const caffe2::NetDef &init)
{
  return OutputShapePtrs(net, init).get_value();
}

/*
 *  Optimizers
 */

// Optimizers all need the same variables, and their workflow are similar.
// Common behavior was put in a class to prevent code duplication
class AbstractOptimizer
{

  // Keep an access to the nets
  const ModelContext &_context;
  caffe2::NetDef &_netdef;
  caffe2::NetDef &_initdef;
  std::map<std::string, std::vector<int>> _shapes;

protected:
  // Members the childs may need
  ScopedNet _net;
  std::string _param;
  std::string _grad, _moment, _meansq; // Blob names
  float _momentum, _decay;             // Configuration

  // Create a new ConstantFill :
  //  - On the main device of the init net
  //  - Broadcasted on every device
  //  - Tagged as 'external input' on the main net
  void broadcast_external_constantfill(const std::string &name,
                                       const std::vector<int> &shape,
                                       float value)
  {
    caffe2::OperatorDef fill;
    ConstantFill(fill, name, shape, value);
    copy_and_broadcast_operator(_context, _initdef, fill);
    add_external_input(_net, name);
  }

  // Same as above, but using current parameter size, and a fill of 0
  void default_fillers(const std::vector<std::string> &names)
  {
    for (const std::string &name : names)
      {
        broadcast_external_constantfill(name, _shapes[_param], 0);
      }
  }

  // Functions supposed to be overloaded

  virtual void init()
  {
  }                            // Code executed only once per net
  virtual void optimize() = 0; // Code executed for each parameter
  virtual bool negative_base_lr()
  {
    return false;
  }
  virtual void set_default_momentum()
  {
  }
  virtual void set_default_decay()
  {
  }

public:
  AbstractOptimizer(const ModelContext &context, caffe2::NetDef &netdef,
                    caffe2::NetDef &initdef, float momentum, float decay)
      : _context(context), _netdef(netdef), _initdef(initdef),
        _net(context.scope_net(netdef)), _momentum(momentum), _decay(decay)
  {
  }

  // Common code
  void run()
  {

    // Set the default configuration
    if (_momentum < 0)
      set_default_momentum();
    if (_decay < 0)
      set_default_decay();

    // Call child's 'negative_base_lr'
    int base_lr_sign = negative_base_lr() ? -1 : 1;

    // Find where are defined the 'base_lr's and force their sign
    size_t base_lr_args = 0;
    for (caffe2::OperatorDef &op : *_netdef.mutable_op())
      {
        if (op.type() == "LearningRate")
          {
            for (caffe2::Argument &arg : *op.mutable_arg())
              {
                if (arg.name() == "base_lr")
                  {
                    arg.set_f(base_lr_sign * std::abs(arg.f()));
                    ++base_lr_args;
                  }
              }
          }
      }

    // There should be exactly one 'LearningRate' operator per net and one
    // base_lr per operator
    CAFFE_ENFORCE(base_lr_args == _context.device_count());

    // Collect net's informations
    std::set<std::string> params;
    std::set<std::string> computed;
    std::string main_prefix = _context.get_prefix(0);
    collect_params(_netdef, params, computed, main_prefix);
    _shapes.clear();
    collect_blob_shapes(_initdef, _shapes, main_prefix);

    // Call child's "init" function
    init();

    for (const std::string &param : params)
      {

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
  template <class C> static Optimizer callback()
  {
    return [](const ModelContext &context, caffe2::NetDef &netdef,
              caffe2::NetDef &initdef, float momentum, float decay) {
      C(context, netdef, initdef, momentum, decay).run();
    };
  }
};

class sgd : public AbstractOptimizer
{
  using AbstractOptimizer::AbstractOptimizer;
  void set_default_momentum()
  {
    _momentum = 0.9f;
  }
  bool negative_base_lr()
  {
    return _momentum == 0.f;
  }
  void init()
  {
    if (_momentum == 0.f)
      {
        broadcast_external_constantfill(blob_one, std::vector<int>({ 1 }), 1);
      }
  }
  void optimize()
  {
    if (_momentum == 0.f)
      {
        WeightedSum(_net, { _param, blob_one, _grad, blob_lr }, _param);
      }
    else
      {
        default_fillers({ _moment });
        MomentumSGDUpdate(_net, _param, _moment, _grad, blob_lr, _momentum);
      }
  }
};

class adagrad : public AbstractOptimizer
{
  using AbstractOptimizer::AbstractOptimizer;
  void set_default_decay()
  {
    _decay = 1.f;
  }
  bool negative_base_lr()
  {
    return true;
  }
  void optimize()
  {
    default_fillers({ _moment });
    Adagrad(_net, _param, _moment, _grad, blob_lr, _decay);
  }
};

class adam : public AbstractOptimizer
{
  using AbstractOptimizer::AbstractOptimizer;
  bool negative_base_lr()
  {
    return true;
  }
  void optimize()
  {
    std::string moment1(_moment + "_1");
    std::string moment2(_moment + "_2");
    default_fillers({ moment1, moment2 });
    Adam(_net, _param, moment1, moment2, _grad, blob_lr, blob_iter);
  }
};

class rmsprop : public AbstractOptimizer
{
  using AbstractOptimizer::AbstractOptimizer;
  void set_default_momentum()
  {
    _momentum = 0.f;
  }
  void set_default_decay()
  {
    _decay = 0.9f;
  }
  void init()
  {
    broadcast_external_constantfill(blob_one, std::vector<int>({ 1 }), 1);
  }
  void optimize()
  {
    default_fillers({ _moment, _meansq });
    RmsProp(_net, _grad, _meansq, _moment, blob_one, _momentum, _decay);
    MomentumSGDUpdate(_net, _param, _moment, _grad, blob_lr, _momentum);
  }
};

#define REGISTER_OPTIMIZER(name)                                              \
  {                                                                           \
#name, AbstractOptimizer::callback < name>()                              \
  }
const std::map<std::string, Optimizer> optimizers
    = { REGISTER_OPTIMIZER(sgd), REGISTER_OPTIMIZER(adagrad),
        REGISTER_OPTIMIZER(adam), REGISTER_OPTIMIZER(rmsprop) };
#undef REGISTER_OPTIMIZER

const Optimizer &get_optimizer(const std::string &name)
{
  const auto it = optimizers.find(name);
  if (it == optimizers.end())
    {
      CAFFE_THROW("Optimizer '", name, "' is not supported.");
    }
  return it->second;
}
}
}
