// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ops.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

namespace dd::ncnn_ops
{
  // CPU, unpacked FP32. Axis counts physical NCNN dimensions from the
  // outside in (C,D,H,W). NCNN dispatches independent batches to the layer.
  class TopK final : public ncnn::Layer
  {
  public:
    int load_param(const ncnn::ParamDict &pd) override
    {
      k = pd.get(0, 0);
      axis = pd.get(1, -1);
      largest = pd.get(2, 1);
      expected_width = pd.get(4, 0);
      expected_stride = pd.get(5, 0);
      outputs = pd.get(3, 2); // 0 values, 1 int32 indices, 2 both
      return k > 0 && (largest == 0 || largest == 1) && outputs >= 0
                     && outputs <= 2
                 ? 0
                 : -1;
    }

    int forward(const std::vector<ncnn::Mat> &bottom,
                std::vector<ncnn::Mat> &top,
                const ncnn::Option &opt) const override
    {
      if (bottom.size() != 1 || top.size() != (outputs == 2 ? 2u : 1u))
        return -1;
      const auto &x = bottom[0];
      int dim = axis < 0 ? x.dims + axis : axis;
      if (x.empty() || x.elempack != 1 || x.elemsize != 4 || dim < 0
          || dim >= x.dims)
        return -1;
      std::array<int, 4> sizes{ x.w, x.h, x.dims == 3 ? x.c : x.d, x.c };
      int physical = x.dims - 1 - dim;
      int width = sizes[physical];
      size_t count = size_t(x.w) * x.h * x.d * x.c;
      if (k > width || count > std::numeric_limits<int>::max())
        return -1;
      // Reshape removes any padding between channels before strided access.
      auto flat = x.reshape(int(count), opt.workspace_allocator);
      if (flat.empty())
        return -100;
      const float *input = flat;
      int stride = std::accumulate(sizes.begin(), sizes.begin() + physical, 1,
                                   std::multiplies<int>());
      if ((expected_width && width != expected_width)
          || (expected_stride && stride != expected_stride))
        return -1;
      for (auto &out : top)
        {
          out.create(int(count / width * k), size_t(4), opt.blob_allocator);
          if (out.empty())
            return -100;
        }
      float *values = outputs == 1 ? nullptr : (float *)top.front().data;
      int *indices = outputs == 0 ? nullptr : (int *)top.back().data;
      std::vector<int> order(width);
      for (size_t base = 0; base < count; base += size_t(width) * stride)
        for (int lane = 0; lane < stride; ++lane)
          {
            auto value = [&](int i)
              { return input[base + size_t(i) * stride + lane]; };
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int a, int b)
                                {
                                  float lhs = value(a), rhs = value(b);
                                  if (std::isnan(lhs) != std::isnan(rhs))
                                    return std::isnan(lhs) == bool(largest);
                                  if (lhs == rhs || std::isnan(lhs))
                                    return a < b;
                                  return largest ? lhs > rhs : lhs < rhs;
                                });
            for (int i = 0; i < k; ++i)
              {
                size_t j = base / width * k + size_t(i) * stride + lane;
                if (values)
                  values[j] = value(order[i]);
                if (indices)
                  indices[j] = order[i];
              }
          }
      sizes[physical] = k;
      for (auto &out : top)
        {
          if (x.dims == 2)
            out = out.reshape(sizes[0], sizes[1], opt.blob_allocator);
          if (x.dims == 3)
            out = out.reshape(sizes[0], sizes[1], sizes[2],
                              opt.blob_allocator);
          if (x.dims == 4)
            out = out.reshape(sizes[0], sizes[1], sizes[2], sizes[3],
                              opt.blob_allocator);
          if (out.empty())
            return -100;
        }
      return 0;
    }

  private:
    int k, axis, largest, outputs, expected_width, expected_stride;
  };
  DEFINE_LAYER_CREATOR(TopK)
}
