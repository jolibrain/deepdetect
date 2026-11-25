// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ops.h"
#include <algorithm>

namespace dd::ncnn_ops
{
  ncnn::Layer *TopK_layer_creator(void *);

  class IndicesToFloat final : public ncnn::Layer
  {
  public:
    IndicesToFloat()
    {
      one_blob_only = true;
    }
    int forward(const ncnn::Mat &ids, ncnn::Mat &out,
                const ncnn::Option &opt) const override
    {
      if (ids.elempack != 1 || ids.elemsize != sizeof(int))
        return -1;
      out.create_like(ids, opt.blob_allocator);
      if (out.empty())
        return -100;
      for (int c = 0; c < ids.c; ++c)
        {
          const int *src = ids.channel(c);
          float *dst = out.channel(c);
          std::copy_n(src, size_t(ids.w) * ids.h * ids.d, dst);
        }
      return 0;
    }
  };
  DEFINE_LAYER_CREATOR(IndicesToFloat)

  class GatherRows final : public ncnn::Layer
  {
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &opt) const override
    {
      if (in.size() != 2 || out.size() != 1)
        return -1;
      const auto &data = in[0], &ids = in[1];
      if (data.dims != 2 || ids.dims != 1 || data.elemsize != 4
          || ids.elemsize != 4 || data.elempack != 1 || ids.elempack != 1)
        return -1;
      out[0].create(data.w, ids.w, size_t(4), opt.blob_allocator);
      if (out[0].empty())
        return -100;
      for (int i = 0; i < ids.w; ++i)
        {
          int index = ((const int *)ids.data)[i];
          if (index < 0 || index >= data.h)
            return -1;
          std::copy_n(data.row(index), data.w, out[0].row(i));
        }
      return 0;
    }
  };
  DEFINE_LAYER_CREATOR(GatherRows)

  // Convert ranked query/class pairs and normalized xyxy boxes to service
  // rows.
  class DetectionRows final : public ncnn::Layer
  {
    int classes;
    int load_param(const ncnn::ParamDict &pd) override
    {
      classes = pd.get(0, 0);
      return classes > 0 ? 0 : -1;
    }
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &opt) const override
    {
      if (in.size() != 3 || out.size() != 1)
        return -1;
      const auto &scores = in[0], &ids = in[1], &boxes = in[2];
      if (scores.dims != 1 || ids.dims != 1 || ids.w != scores.w
          || boxes.dims != 2 || boxes.w != 4)
        return -1;
      for (const auto &x : in)
        if (x.elempack != 1 || x.elemsize != 4)
          return -1;
      out[0].create(6, ids.w, size_t(4), opt.blob_allocator);
      if (out[0].empty())
        return -100;
      for (int i = 0; i < ids.w; ++i)
        {
          int id = ((const int *)ids.data)[i];
          if (id < 0 || id / classes >= boxes.h)
            return -1;
          float *row = out[0].row(i);
          row[0] = 1 + id % classes;
          row[1] = scores[i];
          std::copy_n(boxes.row(id / classes), 4, row + 2);
        }
      return 0;
    }
  };
  DEFINE_LAYER_CREATOR(DetectionRows)

  void register_layers(ncnn::Net &net)
  {
    net.register_custom_layer("DeepDetectTopK", TopK_layer_creator);
    net.register_custom_layer("DeepDetectIndicesToFloat",
                              IndicesToFloat_layer_creator);
    net.register_custom_layer("DeepDetectGatherRows",
                              GatherRows_layer_creator);
    net.register_custom_layer("DeepDetectDetectionRows",
                              DetectionRows_layer_creator);
  }
}
