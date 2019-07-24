#include "torchinputconns.h"

namespace dd {

using namespace torch;


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


void TxtTorchInputFileConn::transform(const APIData &ad) {
    try
    {
        TxtInputFileConn::transform(ad);
    }
    catch(const std::exception& e)
    {
        throw;
    }

    if (ad.has("parameters") && ad.getobj("parameters").has("input"))
    {
        APIData ad_input = ad.getobj("parameters").getobj("input");
        if (ad_input.has("width"))
            _width = ad_input.get("width").get<int>();
    }

    if (!_ordered_words || _characters)
        throw InputConnectorBadParamException("Need ordered_words = true with backend torch");

    int cls_pos = _vocab.at("[CLS]")._pos;
    int sep_pos = _vocab.at("[SEP]")._pos;
    int unk_pos = _vocab.at("[UNK]")._pos;

    for (auto *te : _txt)
    {
        TxtOrderedWordsEntry *tow = static_cast<TxtOrderedWordsEntry *>(te);
        tow->reset();

        std::vector<int64_t> ids;
        ids.push_back(cls_pos);

        while(tow->has_elt())
        {
            std::string word;
            double val;
            tow->get_next_elt(word, val);
            std::unordered_map<std::string,Word>::iterator it;
                
            if ((it = _vocab.find(word)) != _vocab.end())
            {
                ids.push_back(it->second._pos);
            }
            else
            {
                ids.push_back(unk_pos);
            }
        }

        ids.push_back(sep_pos);

        at::Tensor ids_tensor = toLongTensor(ids);
        at::Tensor mask_tensor = torch::ones_like(ids_tensor);
        at::Tensor token_type_ids_tensor = torch::zeros_like(ids_tensor);

        int64_t padding_size = _width - ids_tensor.sizes().back();
        ids_tensor = torch::constant_pad_nd(
            ids_tensor, at::IntList{0, padding_size}, 0);
        mask_tensor = torch::constant_pad_nd(
            mask_tensor, at::IntList{0, padding_size}, 0);
        token_type_ids_tensor = torch::constant_pad_nd(
            token_type_ids_tensor, at::IntList{0, padding_size}, 0);

        _dataset.add_batch({ids_tensor, token_type_ids_tensor, mask_tensor}, {});
    }
}

}