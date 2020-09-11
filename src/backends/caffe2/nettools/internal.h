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

#ifndef CAFFE2NETTOOLSINTERNAL_H
#define CAFFE2NETTOOLSINTERNAL_H

#include "backends/caffe2/nettools.h"

namespace dd
{
  namespace Caffe2NetTools
  {

    /*
     * Operators that don't have a gradient. The associated boolean is set to
     * true for filling operators (only used to initialize blobs) and false for
     * other operator types
     */
    extern const std::map<std::string, bool> non_trainable_ops;

    /*
     * Gives informations about the training behavior of a given operator
     */
    class TrainableOp
    {
    protected:
      /*
       * Describes the properties of a specific output of the operator
       */
      class Output
      {
      public:
        bool _training_only;  // Should not exist in prediction
        bool _based_on_input; // Whether the name is based on an input or an
                              // output
        int _based_on_index;  // Index of the blob used to generate the name
        std::string _suffix;  // String appended to the base name
        int _can_overwrite;   // Index of the input it can overwrite (< 0 means
                              // none)
        Output(bool training, bool input, int index, const std::string &suffix,
               int overwrite)
            : _training_only(training), _based_on_input(input),
              _based_on_index(index), _suffix(suffix),
              _can_overwrite(overwrite)
        {
        }
      };

      const std::set<int>
          _computed_inputs; // Indices of 'computed parameter' tagged inputs
      const std::vector<Output> _outputs;
      TrainableOp(const std::set<int> &inputs,
                  const std::vector<Output> &outputs)
          : _computed_inputs(inputs), _outputs(outputs)
      {
      }

    public:
      TrainableOp() : TrainableOp({}, {})
      {
      }

      /*
       * Inputs that will be part of the gradient
       */
      void get_trainable_inputs(const caffe2::OperatorDef &op,
                                std::set<std::string> &inputs) const;

      /*
       * Inputs that should be treated as 'computed parameters'
       *
       * Quote from 'model_helper.py':
       *
       *       'Computed params' are such parameters that are not optimized via
       * gradient descent but are directly computed from data, such as the
       * running mean and variance of Spatial Batch Normalization.
       *
       * See 'ParameterTags' in
       * pytorch/caffe2/python/modeling/parameter_info.py
       */
      void get_computed_inputs(const caffe2::OperatorDef &op,
                               std::set<std::string> &inputs) const;

      /*
       * Outputs that must be added (in the given order) for the operator to be
       * trainable.
       *
       * The following quote from the SpatialBN documentation:
       *
       *        Output 3 'var' :
       *        The running variance after the spatial BN operator.
       *        Must be in-place with the input var.
       *        Should not be used for testing.
       *
       * Means that this output won't be present in prediciton, but must be set
       * to a specific value when training.
       *
       * (See
       * https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc
       *  or
       * https://github.com/pytorch/pytorch/blob/master/caffe2/operators/instance_norm_op.cc
       *  for concrete examples)
       */
      void get_needed_outputs(const caffe2::OperatorDef &op,
                              std::vector<std::string> &outputs) const;

      /*
       * Outputs that overwrite an input while keeping the operator trainable
       */
      void get_trainable_overwrites(const caffe2::OperatorDef &op,
                                    std::set<int> &outputs) const;
    };

    /*
     * Operators having a gradient.
     */
    extern const std::map<std::string, TrainableOp> trainable_ops;

    extern const std::string batch_splits_suffix;
    extern const std::string force_device_suffix;
    extern const std::string mean_square_suffix;
    extern const std::string momentum_suffix;
    extern const std::string gradient_suffix;
    extern const std::string scale_suffix;

    extern const std::string blob_lr;
    extern const std::string blob_one;
    extern const std::string blob_iter;
    extern const std::string blob_xent;
    extern const std::string blob_loss;
    extern const std::string blob_loss_scale;

    /*
     * Checks if the operator have a gradient and store a pointer to the
     * 'TrainableOp' if any
     */
    bool is_trainable(const caffe2::OperatorDef &op,
                      const TrainableOp *&train_op);
  }
}

#endif
