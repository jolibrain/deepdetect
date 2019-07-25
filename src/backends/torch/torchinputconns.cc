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

    if (!_ordered_words || _characters)
        throw InputConnectorBadParamException("Need ordered_words = true");

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
        /* // Exemple in:
        {
            101, 2489, 4443, 1999, 1016, 1037, 1059, 2243, 2135, 4012, 2361,
            2000, 2663, 6904, 2452, 2345, 1056, 25509, 2015, 7398, 2089, 2384,
            1012, 3793, 6904, 2000, 6584, 12521, 2487, 2000, 4374, 4443, 3160,
            1006, 2358, 2094, 19067, 2102, 3446, 1007, 1056, 1004, 1039, 1005,
            1055, 6611, 5511, 19961, 22407, 18613, 23352, 7840, 15136, 1005, 1055, 102
        }; */

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

        int64_t padding_size = _in_size - ids_tensor.sizes().back();
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