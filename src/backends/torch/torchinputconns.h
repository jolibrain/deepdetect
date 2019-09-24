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
    typedef torch::data::Example<std::vector<at::Tensor>, std::vector<at::Tensor>> TorchBatch;

    class TorchDataset : public torch::data::BatchDataset
        <TorchDataset, c10::optional<TorchBatch>>
    {
    private:
        bool _shuffle = false;
        long _seed = -1;
        std::vector<int64_t> _indices;

    public:
        /// Vector containing the whole dataset (the "cached data").
        std::vector<TorchBatch> _batches;


        TorchDataset() {}

        void add_batch(std::vector<at::Tensor> data, std::vector<at::Tensor> target = {});

        void reset();

        /// Size of data loaded in memory
        size_t cache_size() const { return _batches.size(); }

        c10::optional<size_t> size() const override {
            return cache_size();
        }

        bool empty() const { return cache_size() == 0; }

        c10::optional<TorchBatch> get_batch(BatchRequestType request) override;

        /// Returns a batch containing all the cached data
        TorchBatch get_cached();

        /// Split a percentage of this dataset
        TorchDataset split(double start, double stop);
    };


    struct MaskedLMParams
    {
        double _change_prob = 0.15; /**< When masked LM learning, probability of changing a token (mask/randomize/keep). */
        double _mask_prob =  0.8; /**< When masked LM learning, probability of masking a token. */
        double _rand_prob = 0.1; /**< When masked LM learning, probability of randomizing a token. */
    };


    class TorchInputInterface
    {
    public:
        TorchInputInterface() {}
        TorchInputInterface(const TorchInputInterface &i)
             : _finetuning(i._finetuning),
             _lm_params(i._lm_params),
             _dataset(i._dataset),
             _test_dataset(i._test_dataset),
             _input_format(i._input_format) { }

        ~TorchInputInterface() {}

        torch::Tensor toLongTensor(std::vector<int64_t> &values) {
            int64_t val_size = values.size();
            return torch::from_blob(&values[0], at::IntList{val_size}, at::kLong).clone();
        }

        TorchBatch generate_masked_lm_batch(const TorchBatch &example) { return {}; }

        int64_t mask_id() const { return 0; }
        int64_t vocab_size() const { return 0; }
        std::string get_word(int64_t id) const { return ""; }


        TorchDataset _dataset;
        TorchDataset _test_dataset;

        MaskedLMParams _lm_params;
        bool _finetuning;
        /** Tell which inputs should be provided to the models.
         * see*/
        std::string _input_format;
        std::vector<int64_t> _lengths;/**< length of each sentence with txt connector. */
    };

    class ImgTorchInputFileConn : public ImgInputFileConn, public TorchInputInterface
    {
    public:
        ImgTorchInputFileConn()
            :ImgInputFileConn() {}
        ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
            :ImgInputFileConn(i),TorchInputInterface(i), _std{i._std} {}
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

            if (ad.has("parameters"))
            {
                APIData ad_param = ad.getobj("parameters");
                if (ad_param.has("input"))
                {
                    APIData input_ad = ad_param.getobj("input");
                    if (input_ad.has("std"))
                        _std = input_ad.get("std").get<double>();
                }
            }

            std::vector<int64_t> sizes{ _height, _width, 3 };
            at::TensorOptions options(at::ScalarType::Byte);

            for (const cv::Mat &bgr : this->_images) {
                at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
                imgt = imgt.toType(at::kFloat).permute({2, 0, 1});
                if (_std != 1.0)
                    imgt = imgt.mul(1. / _std);
                _dataset.add_batch({imgt});
            }
        }

    public:
        at::Tensor _in;

        double _std = 1.0;
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

        void init(const APIData &ad)
        {
            TxtInputFileConn::init(ad);
            fillup_parameters(ad);
        }

        void fillup_parameters(const APIData &ad_input);

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

        int64_t mask_id() const { return _mask_id; }

        int64_t vocab_size() const { return _vocab.size(); }

        std::string get_word(int64_t id) const {
            return _inv_vocab.at(id);
        }

        void transform(const APIData &ad);

        TorchBatch generate_masked_lm_batch(const TorchBatch &example);

        void fill_dataset(TorchDataset &dataset, const std::vector<TxtEntry<double>*> &entries);
    public:
        /** width of the input tensor */
        int _width = 512;
        int _height = 0;
        std::mt19937 _rng;
        /// token id to vocabulary word
        std::map<int, std::string> _inv_vocab;

        int64_t _mask_id = -1; /**< ID of mask token in the vocabulary. */
        int64_t _cls_pos = -1;
        int64_t _sep_pos = -1;
        int64_t _unk_pos = -1;
        int64_t _eot_pos = -1; /**< end of text */


        void make_inv_vocab() {
            _inv_vocab.clear();

            for (auto &entry : _vocab)
            {
                _inv_vocab[entry.second._pos] = entry.first;
            }
        }
    };
} // namespace dd

#endif // TORCHINPUTCONNS_H
