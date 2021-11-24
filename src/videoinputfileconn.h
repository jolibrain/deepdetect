/**
 * DeepDetect
 * Copyright (c) 2021 Emmanuel Benazera
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef VIDEOINPUTFILECONN_H
#define VIDEOINPUTFILECONN_H

#include "dto/input_connector.hpp"

namespace dd
{
}

#ifdef USE_CAFFE
#include "caffeinputconns.h"
#endif

#ifdef USE_TF
#include "backends/tf/tfinputconns.h"
#endif

#ifdef USE_DLIB
#include "backends/dlib/dlibinputconns.h"
#endif

#ifdef USE_NCNN
#include "backends/ncnn/ncnninputconns.h"
#endif

#ifdef USE_CAFFE2
#include "backends/caffe2/caffe2inputconns.h"
#endif

#ifdef USE_TENSORRT
#include "backends/tensorrt/tensorrtinputconns.h"
#endif

#ifdef USE_TORCH
#include "backends/torch/torchinputconns.h"
#endif

#endif
