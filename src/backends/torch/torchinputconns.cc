#include "torchinputconns.h"

namespace dd {

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

    std::vector<at::Tensor> vids;
    std::vector<at::Tensor> vmask;

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
        // at::Tensor token_type_ids_tensor = torch::zeros_like(ids_tensor);

        int64_t padding_size = _width - ids_tensor.sizes().back();
        ids_tensor = torch::constant_pad_nd(
            ids_tensor, at::IntList{0, padding_size}, 0);
        mask_tensor = torch::constant_pad_nd(
            mask_tensor, at::IntList{0, padding_size}, 0);
        // token_type_ids_tensor = torch::constant_pad_nd(
        //    token_type_ids_tensor, at::IntList{0, padding_size}, 0);

        vids.push_back(ids_tensor);
        vmask.push_back(mask_tensor);
    }

    _in = torch::stack(vids, 0);
    _attention_mask = torch::stack(vmask, 0);
}

}