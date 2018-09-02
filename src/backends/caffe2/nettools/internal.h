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

namespace dd {
  namespace Caffe2NetTools {

    /*
     * Operators that don't have a gradient. The associated boolean is set to true for
     * filling operators (only used to initialize blobs) and false for other operator types
     */
    extern const std::map<std::string, bool> non_trainable_ops;

    /*
     * Operators having a gradient.
     * With each operator is associated three functions that can store the following :
     *
     * -- 1 --
     *
     * Inputs that will be part of the gradient
     *
     * -- 2 --
     *
     * Inputs that should be treated as 'computed parameters'
     *
     * Quote from 'model_helper.py':
     *
     *       'Computed params' are such parameters that are not optimized via gradient descent
     *       but are directly computed from data, such as the running mean and variance of
     *       Spatial Batch Normalization.
     *
     * See 'ParameterTags' in pytorch/caffe2/python/modeling/parameter_info.py
     *
     * -- 3 --
     *
     * Outputs that must be added (in the given order) for the operator to be trainable.
     *
     * The following quote from the SpatialBN documentation:
     *
     *        Output 3 'var' :
     *        The running variance after the spatial BN operator.
     *        Must be in-place with the input var.
     *        Should not be used for testing.
     *
     * Means that this output won't be present in prediciton, but must be set to a specific value
     * when training.
     *
     * (See https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_batch_norm_op.cc
     *  or https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc
     *  for concrete examples)
     *
     */
    using GetOpInputFct =
      std::function<void(const caffe2::OperatorDef&, std::set<std::string>&)>;
    using GetOpOutputFct =
      std::function<void(const caffe2::OperatorDef&, std::vector<std::string>&)>;
    using GetOpBlobsFcts = std::tuple<GetOpInputFct, GetOpInputFct, GetOpOutputFct>;
    extern const std::map<std::string, GetOpBlobsFcts> trainable_ops;

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

  }
}

#endif
