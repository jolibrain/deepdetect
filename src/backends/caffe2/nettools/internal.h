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
     * With each operator is associated a list of indices,
     * corresponding to the inputs that should be treated as 'computed parameters'
     * ( A 'computed parameter' act almost like a weight or a bias,
     *   but is not changed by the gradients. See ParameterTags in
     *   pytorch/caffe2/python/modeling/parameter_info.py )
     */
    extern const std::map<std::string, std::set<int> > trainable_ops;

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
