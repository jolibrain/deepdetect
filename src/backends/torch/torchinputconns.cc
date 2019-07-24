#include "torchinputconns.h"

namespace dd {

using namespace torch;

// ===== TorchDataset

void TorchDataset::add_batch(std::vector<at::Tensor> data, std::vector<at::Tensor> target)
{
    _batches.push_back(TorchBatch(data, target));
}

void TorchDataset::reset()
{
    _indices.clear();

    for (int64_t i = 0; i < _batches.size(); ++i) {
        _indices.push_back(i);
    }

    if (_shuffle)
    {
        auto seed = _seed == -1 ? static_cast<long>(time(NULL)) : _seed;
        std::shuffle(_indices.begin(), _indices.end(), std::mt19937(seed));
    }
}

// `request` holds the size of the batch
// Data selection and batch construction are done in this method
c10::optional<TorchBatch> TorchDataset::get_batch(BatchRequestType request)
{
    size_t count = request[0];
    count = count < _indices.size() ? count : _indices.size();

    if (count == 0) {
        return torch::nullopt;
    }

    std::vector<std::vector<Tensor>> data, target;

    while(count != 0) {
        auto id = _indices.back();
        auto entry = _batches[id];

        for (int i = 0; i < entry.data.size(); ++i)
        {
            while (i >= data.size())
                data.emplace_back();
            data[i].push_back(entry.data.at(i));
        }
        for (int i = 0; i < entry.target.size(); ++i)
        {
            while (i >= target.size())
                target.emplace_back();
            target[i].push_back(entry.target.at(i));
        }

        _indices.pop_back();
        count--;
    }

    std::vector<Tensor> data_tensors;
    for (auto vec : data)
        data_tensors.push_back(torch::stack(vec));

    std::vector<Tensor> target_tensors;
    for (auto vec : target)
        target_tensors.push_back(torch::stack(vec));

    return TorchBatch{ data_tensors, target_tensors };
}

TorchBatch TorchDataset::get_cached() {
    reset();
    auto batch = get_batch({cache_size()});
    if (!batch)
        throw InputConnectorInternalException("No data provided");
    return batch.value();
}

TorchDataset TorchDataset::split(double start, double stop)
{
    auto datasize = _batches.size();
    auto start_it = _batches.begin() + static_cast<int64_t>(datasize * start);
    auto stop_it = _batches.end() - static_cast<int64_t>(datasize * (1 - stop));

    TorchDataset new_dataset;
    new_dataset._batches.insert(new_dataset._batches.end(), start_it, stop_it);
    return new_dataset;
}


// ===== TxtTorchInputFileConn

void TxtTorchInputFileConn::fillup_parameters(const APIData &ad_input)
{
  TxtInputFileConn::fillup_parameters(ad_input);
}

void TxtTorchInputFileConn::transform(const APIData &ad) {
    // if (_finetuning)
    // XXX: Generating vocab from scratch is not currently
    _generate_vocab = false;

    try
    {
        TxtInputFileConn::transform(ad);
    }
    catch(const std::exception& e)
    {
        throw;
    }

    if (!_ordered_words || _characters)
        throw InputConnectorBadParamException("Need ordered_words = true with backend torch");

    // XXX: move in txtinputconn?
    make_inv_vocab();

    if (ad.has("parameters") && ad.getobj("parameters").has("input"))
    {
        APIData ad_input = ad.getobj("parameters").getobj("input");
        fillup_parameters(ad_input);
    }
    
    if (_input_format == "bert")
    {
        _cls_pos = _vocab.at("[CLS]")._pos;
        _sep_pos = _vocab.at("[SEP]")._pos;
        _unk_pos = _vocab.at("[UNK]")._pos;
        _mask_id = _vocab.at("[MASK]")._pos;
    }
    else if (_input_format == "gpt2")
    {
        _eot_pos = _vocab.at("<|endoftext|>")._pos;
    }

    fill_dataset(_dataset, _txt);
    if (!_test_txt.empty())
        fill_dataset(_test_dataset, _test_txt);
}

TorchBatch TxtTorchInputFileConn::generate_masked_lm_batch(const TorchBatch &example)
{
    std::uniform_real_distribution<double> uniform(0, 1);
    std::uniform_int_distribution<int64_t> vocab_distrib(0, vocab_size() - 1);
    Tensor input_ids = example.data.at(0).clone();
    // lm_labels: n_batch * sequence_length
    // equals to input_ids where tokens are masked, and -1 otherwise
    Tensor lm_labels = torch::ones_like(input_ids, TensorOptions(kLong)) * -1;

    // mask random tokens
    auto input_acc = input_ids.accessor<int64_t,2>();
    auto att_mask_acc = example.data.at(2).accessor<int64_t,2>();
    auto labels_acc = lm_labels.accessor<int64_t,2>();
    for (int i = 0; i < input_ids.size(0); ++i)
    {
        int j = 1; // skip [CLS] token
        while (j < input_ids.size(1) && att_mask_acc[i][j] != 0)
        {
            double rand_num = uniform(_rng);
            if (rand_num < _lm_params._change_prob && input_acc[i][j] != _sep_pos)
            {
                labels_acc[i][j] = input_acc[i][j];

                rand_num = uniform(_rng);
                if (rand_num < _lm_params._mask_prob)
                {
                    input_acc[i][j] = mask_id();
                }
                else if (rand_num < _lm_params._mask_prob + _lm_params._rand_prob)
                {
                    input_acc[i][j] = vocab_distrib(_rng);
                }
            }
            ++j;
        }
    }

    TorchBatch output;
    output.target.push_back(lm_labels);
    output.data.push_back(input_ids);
    for (int i = 1; i < example.data.size(); ++i)
    {
        output.data.push_back(example.data[i]);
    }
    return output;
}

void TxtTorchInputFileConn::fill_dataset(TorchDataset &dataset,
                                         const std::vector<TxtEntry<double>*> &entries)
{
    for (auto *te : entries)
    {
        TxtOrderedWordsEntry *tow = static_cast<TxtOrderedWordsEntry *>(te);
        tow->reset();
        std::string word;
        double val;
        std::vector<int64_t> ids;

        while(tow->has_elt())
        {
            if (ids.size() >= _width)
                break;

            tow->get_next_elt(word, val);
            std::unordered_map<std::string,Word>::iterator it;

            if ((it = _vocab.find(word)) != _vocab.end())
            {
                ids.push_back(it->second._pos);
            }
            else if (_input_format == "bert")
            {
                ids.push_back(_unk_pos);
            }
        }

        // Extract last token (needed by gpt2)
        int64_t last_token = 0;
        if (tow->has_elt())
        {
            tow->get_next_elt(word, val);
            std::unordered_map<std::string,Word>::iterator it;

            if ((it = _vocab.find(word)) != _vocab.end())
                last_token = it->second._pos;
        }

        // Post-processing for each model
        if (_input_format == "bert")
        {
            // make room for cls and sep token
            while (ids.size() > _width - 2)
                ids.pop_back();

            ids.insert(ids.begin(), _cls_pos);
            ids.push_back(_sep_pos);
        }
        else if (_input_format == "gpt2")
        {
            if (ids.size() < _width)
            {
                ids.push_back(_eot_pos);
            }
        }

        at::Tensor ids_tensor = toLongTensor(ids);
        at::Tensor mask_tensor = torch::ones_like(ids_tensor);
        at::Tensor token_type_ids_tensor = torch::zeros_like(ids_tensor);

        int64_t seq_len = ids_tensor.sizes().back();
        int64_t padding_size = _width - seq_len;
        _lengths.push_back(seq_len);
        ids_tensor = torch::constant_pad_nd(
            ids_tensor, at::IntList{0, padding_size}, 0);
        mask_tensor = torch::constant_pad_nd(
            mask_tensor, at::IntList{0, padding_size}, 0);
        token_type_ids_tensor = torch::constant_pad_nd(
            token_type_ids_tensor, at::IntList{0, padding_size}, 0);
        at::Tensor position_ids = torch::arange(_width, at::kLong);

        std::vector<Tensor> target_vec;
        int target_val = static_cast<int>(tow->_target);

        if (target_val != -1)
        {
            Tensor target_tensor = torch::full(1, target_val, torch::kLong);
            target_vec.push_back(target_tensor);
        }

        if (_input_format == "bert")
            dataset.add_batch({ids_tensor, token_type_ids_tensor, mask_tensor}, std::move(target_vec));
        else if (_input_format == "gpt2")
        {
            std::vector<Tensor> out_vec { ids_tensor.slice(0, 1) };
            out_vec.push_back(torch::full(1, last_token, torch::kLong));
            target_vec.insert(target_vec.begin(), torch::cat(out_vec, 0));
            dataset.add_batch({ids_tensor, position_ids}, std::move(target_vec));
        }
    }
}

}
