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

#ifndef SEGMENTMASKOP_H
#define SEGMENTMASKOP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

  template <class Context>
    class SegmentMaskOp final : public Operator<Context> {
  public:

    USE_OPERATOR_CONTEXT_FUNCTIONS;
    SegmentMaskOp(const OperatorDef &op, Workspace *ws): Operator<Context>(op, ws),
      _thresh_bin(OperatorBase::template GetSingleArgument<float>("thresh_bin", 0.5)) {}

    bool RunOnDevice() override;

  private:
    float _thresh_bin;
  };
}

#endif
