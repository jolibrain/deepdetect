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

#ifndef BBOXTOROIOP_H
#define BBOXTOROIOP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

  template <class Context>
    class BBoxToRoiOp final : public Operator<Context> {
  public:

    USE_OPERATOR_CONTEXT_FUNCTIONS;
    BBoxToRoiOp(const OperatorDef &op, Workspace *ws): Operator<Context>(op, ws) {}

    bool RunOnDevice() override;
  };
}

#endif
