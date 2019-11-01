/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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

#ifndef TORCHINPUTCONNS_H
#define TORCHINPUTCONNS_H

#include <vector>

#include <torch/torch.h>

#include "imginputfileconn.h"
#include "txtinputfileconn.h"

namespace dd
{
    class TorchInputInterface
    {
    public:
        TorchInputInterface() {}
        TorchInputInterface(const TorchInputInterface &i)
            : _in(i._in), _attention_mask(i._attention_mask) {}

        ~TorchInputInterface() {}

        torch::Tensor toLongTensor(std::vector<int64_t> &values) {
            int64_t val_size = values.size();
            return torch::from_blob(&values[0], at::IntList{val_size}, at::kLong);
        }

        at::Tensor _in;
        at::Tensor _attention_mask;
    };

    class ImgTorchInputFileConn : public ImgInputFileConn, public TorchInputInterface
    {
    public:
        ImgTorchInputFileConn()
            :ImgInputFileConn() {}
        ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
            :ImgInputFileConn(i),TorchInputInterface(i) {}
        ~ImgTorchInputFileConn() {}

        // for API info only
        int width() const
        {
            return _width;
        }

        // for API info only
        int height() const
        {
            return _height;
        }

        void init(const APIData &ad)
        {
            ImgInputFileConn::init(ad);
        }
        
        void transform(const APIData &ad)
        {
            try
            {
                ImgInputFileConn::transform(ad);
            }
            catch(const std::exception& e)
            {
                throw;
            }
	    
            std::vector<at::Tensor> tensors;
            std::vector<int64_t> sizes{ _height, _width, 3 };
            at::TensorOptions options(at::ScalarType::Byte);

            for (const cv::Mat &bgr : this->_images) {
                at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
                imgt = imgt.toType(at::kFloat).permute({2, 0, 1});
		size_t nchannels = imgt.size(0);
		if (_scale != 1.0)
		  imgt = imgt.mul(_scale);
		if (!_mean.empty() && _mean.size() != nchannels)
		  throw InputConnectorBadParamException("mean vector be of size the number of channels (" + std::to_string(nchannels) + ")");
		for (size_t m=0;m<_mean.size();m++)
		  imgt[0][m] = imgt[0][m].sub_(_mean.at(m));
		if (!_std.empty() && _std.size() != nchannels)
		  throw InputConnectorBadParamException("std vector be of size the number of channels (" + std::to_string(nchannels) + ")");
		for (size_t s=0;s<_std.size();s++)
		  imgt[0][s] = imgt[0][s].div_(_std.at(s));
                tensors.push_back(imgt);
            }

            _in = torch::stack(tensors, 0);
        }

    public:
        at::Tensor _in;
    };


    class TxtTorchInputFileConn : public TxtInputFileConn, public TorchInputInterface
    {
    public:
        TxtTorchInputFileConn()
            : TxtInputFileConn() {
            _vocab_sep = '\t';
        }
        TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
            : TxtInputFileConn(i), TorchInputInterface(i),
              _width(i._width), _height(i._height) {}
        ~TxtTorchInputFileConn() {}

        // for API info only
        int width() const
        {
            return _width;
        }

        // for API info only
        int height() const
        {
            return _height;
        }

        void transform(const APIData &ad);

    public:
        int _width = 512;
        int _height = 0;
    };
} // namespace dd

#endif // TORCHINPUTCONNS_H
