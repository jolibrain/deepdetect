/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERINPUTCONNS_H
#define PYTORCHWORKERINPUTCONNS_H

#include "imginputfileconn.h"

namespace dd
{
  class ImgPytorchInputFileConn : public ImgInputFileConn
  {
  public:
    ImgPytorchInputFileConn() : ImgInputFileConn()
    {
    }

    ImgPytorchInputFileConn(const ImgPytorchInputFileConn &other)
        : ImgInputFileConn(other)
    {
    }

    int width() const
    {
      return _width;
    }

    int height() const
    {
      return _height;
    }
  };
}

#endif
