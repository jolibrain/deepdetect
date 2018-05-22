/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef CAFFE2INPUTCONNS_H
#define CAFFE2INPUTCONNS_H

//TODO Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/core/tensor.h>
#pragma GCC diagnostic pop

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"
#include "svminputfileconn.h"

namespace dd
{
  /**
   * \brief high-level data structure shared among Caffe2-compatible connectors of DeepDetect
   */
  class Caffe2InputInterface
  {
  public:
    Caffe2InputInterface() {}
    ~Caffe2InputInterface() {}

      /**
       * \brief fill the input tensor
       * @param tensor the tensor to fill
       * @param num the batch size
       * @return the real size of this batch (<= num)
       * @see ImgCaffe2InputFileConn
       */

    int get_tensor_test(caffe2::TensorCPU &, int) {
      return 0;
    }

    std::vector<std::string> _ids; /* input ids (e.g. image ids) */
  };

  /**
   * \brief Caffe2 image connector
   */
  class ImgCaffe2InputFileConn : public ImgInputFileConn, public Caffe2InputInterface
  {
  public:
  ImgCaffe2InputFileConn()
    :ImgInputFileConn(), Caffe2InputInterface() {
    }
    ~ImgCaffe2InputFileConn() {}

    void init(const APIData &ad);
    void transform(const APIData &ad);

    int get_tensor_test(caffe2::TensorCPU &tensor, int num);
  };
}

#endif
