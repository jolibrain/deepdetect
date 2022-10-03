/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#include "simsearch.h"
#include "utils/fileops.hpp"
#include "utils/utils.hpp"
#ifdef USE_FAISS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "faiss/IndexIVF.h"
#pragma GCC diagnostic pop
#include "faiss/invlists/OnDiskInvertedLists.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/index_factory.h"
#ifdef USE_GPU_FAISS
#include "faiss/gpu/GpuCloner.h"
#endif
#endif

namespace dd
{

  /*-- URIData --*/
  char URIData::_enc_char = '^';

  std::string URIData::encode() const
  {
    if (_bbox.empty())
      return _uri;
    std::string enc = _uri + _enc_char;
    for (auto b : _bbox)
      enc += std::to_string(b) + _enc_char;
    enc += std::to_string(_prob) + _enc_char;
    enc += _cat;
    return enc;
  }

  void URIData::decode(const std::string &tmp)
  {
    std::vector<std::string> tok = dd_utils::split(tmp, _enc_char);
    _uri = tok.at(0);
    if (tok.size() == 1)
      return;
    for (int i = 1; i < 5; i++) // bbox is 4 double
      {
        _bbox.push_back(std::stod(tok.at(i)));
      }
    _prob = std::stod(tok.at(5));
    _cat = tok.at(6);
  }

  /*-- SearchEngine --*/
  template <class TSE>
  SearchEngine<TSE>::SearchEngine(const int &dim,
                                  const std::string &model_repo)
      : _dim(dim)
  {
    _tse = new TSE(_dim, model_repo);
  }

  template <class TSE> SearchEngine<TSE>::~SearchEngine()
  {
    delete _tse;
  }

  template <class TSE> void SearchEngine<TSE>::create_index()
  {
    _tse->create_index();
  }

  template <class TSE> void SearchEngine<TSE>::update_index()
  {
    _tse->update_index();
  }

  template <class TSE> void SearchEngine<TSE>::remove_index()
  {
    std::cerr << "removing index\n";
    _tse->remove_index();
  }

  template <class TSE>
  void SearchEngine<TSE>::index(const URIData &uri,
                                const std::vector<double> &data)
  {
    std::lock_guard<std::mutex> lock(_index_mutex);
    _tse->index(uri, data);
  }

  template <class TSE>
  void SearchEngine<TSE>::index(const std::vector<URIData> &uris,
                                const std::vector<std::vector<double>> &datas)
  {
    std::lock_guard<std::mutex> lock(_index_mutex);
    _tse->index(uris, datas);
  }

  template <class TSE>
  void SearchEngine<TSE>::search(const std::vector<double> &data,
                                 const int &nn, std::vector<URIData> &uris,
                                 std::vector<double> &distances)
  {
    _tse->search(data, nn, uris, distances);
  }

#ifdef USE_ANNOY
  /*- AnnoySE -*/

  AnnoySE::AnnoySE(const int &f, const std::string &model_repo)
      : _f(f), _model_repo(model_repo)
  {
    _aindex = new AnnoyIndex<int, double, Angular, Kiss32Random,
                             AnnoyIndexSingleThreadedBuildPolicy>(f);
    _db = db::GetDB(_db_backend);
  }

  AnnoySE::~AnnoySE()
  {
    delete _aindex;
    if (_db)
      _db->Close();
    delete _db;
  }

  void AnnoySE::create_index() // TODO: exception
  {
    std::string index_filename = _model_repo + "/" + _index_name;
    if (fileops::file_exists(index_filename))
      {
        _saved_tree = true;
        _aindex->load(index_filename.c_str(), _map_populate);
        _built_index = true;
      }
    std::string db_filename = _model_repo + "/" + _db_name;
    if (fileops::file_exists(db_filename))
      {
        std::cerr << "open existing index db\n";
        _db->Open(db_filename, db::WRITE);
      }
    else
      {
        std::cerr << "create index db\n";
        _db->Open(db_filename, db::NEW);
      }
  }

  void AnnoySE::remove_index()
  {
    fileops::remove_file(_model_repo, _index_name);
    std::string db_filename = _model_repo + "/" + _db_name;
    fileops::clear_directory(db_filename);
    rmdir(db_filename.c_str());
  }

  void AnnoySE::update_index()
  {
    build_tree();
    save_tree(); // no turning back
  }

  void AnnoySE::build_tree()
  {
    if (_count_put % _count_put_max != 0)
      {
        _txn->Commit(); // last pending db commit
        _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
      }
    _aindex->build(_ntrees);
    _built_index = true;
  }

  void AnnoySE::unbuild_tree()
  {
    _aindex->unbuild();
    _built_index = false;
  }

  void AnnoySE::save_tree()
  {
    std::string index_path = _model_repo + "/" + _index_name;
    _aindex->save(index_path.c_str(), _map_populate);
    _saved_tree = true;
  }

  // must be protected by mutex
  void AnnoySE::index(const URIData &uri, const std::vector<double> &vec)
  {
    if (_saved_tree)
      throw SimIndexException("Cannot index after Annoy index has been saved");
    int idx = _index_size;
    _aindex->add_item(idx, &vec[0]);
    ++_index_size;
    add_to_db(idx, uri);
  }

  void AnnoySE::index(const std::vector<URIData> &uris,
                      const std::vector<std::vector<double>> &vecs)
  {
    for (size_t i = 0; i < uris.size(); ++i)
      {
        index(uris[i], vecs[i]);
      }
  }

  void AnnoySE::search(const std::vector<double> &vec, const int &nn,
                       std::vector<URIData> &uris,
                       std::vector<double> &distances)
  {
    if (!_built_index)
      throw SimSearchException(
          "Cannot search before the Annoy tree has been built");
    std::vector<int> result;
    _aindex->get_nns_by_vector(&vec[0], nn, -1, &result, &distances);
    for (auto i : result)
      {
        URIData uri;
        get_from_db(i, uri);
        uris.push_back(uri);
      }
  }

  void AnnoySE::add_to_db(const int &idx, const URIData &fmap)
  {
    if (_count_put == 0)
      _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
    _txn->Put(std::to_string(idx), fmap.encode());
    ++_count_put;
    if (_count_put % _count_put_max == 0)
      {
        _txn->Commit(); // batch commit
        _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
      }
  }

  void AnnoySE::get_from_db(const int &idx, URIData &fmap)
  {
    std::string tmp;
    _db->Get(std::to_string(idx), tmp);
    fmap.decode(tmp);
  }

  template class SearchEngine<AnnoySE>;
#endif

#ifdef USE_FAISS
  FaissSE::FaissSE(const int &f, const std::string &model_repo)
      : _f(f), _model_repo(model_repo)
  {
    _db = db::GetDB(_db_backend);
    _index_key = std::string("Flat");
  }

  FaissSE::~FaissSE()
  {
    delete _findex;
    if (_db)
      _db->Close();
    delete _db;
  }

  void FaissSE::create_index()
  {
    if (_findex)
      delete _findex;
    std::string index_filename = _model_repo + "/" + _index_name;
    if (fileops::file_exists(index_filename))
      {
        if (_ondisk)
          _findex = faiss::read_index(index_filename.c_str());
        else
          _findex = faiss::read_index(index_filename.c_str());
        _index_size = _findex->ntotal;
      }
    else
      {
        _findex = faiss::index_factory(_f, _index_key.c_str());
        if (_ondisk)
          {
            std::string odilfn = _model_repo + "/" + _il_name;
            faiss::IndexIVF *iivf = dynamic_cast<faiss::IndexIVF *>(_findex);
            if (iivf)
              {
                faiss::OnDiskInvertedLists *odil
                    = new faiss::OnDiskInvertedLists(
                        iivf->nlist, iivf->code_size, odilfn.c_str());
                iivf->own_invlists = true;
                iivf->replace_invlists(odil, true);
              }
            else
              {
                faiss::IndexPreTransform *ipivf
                    = dynamic_cast<faiss::IndexPreTransform *>(_findex);
                if (ipivf)
                  {
                    faiss::IndexIVF *iivf
                        = dynamic_cast<faiss::IndexIVF *>(ipivf->index);
                    if (iivf)
                      {
                        faiss::OnDiskInvertedLists *odil
                            = new faiss::OnDiskInvertedLists(
                                iivf->nlist, iivf->code_size, odilfn.c_str());
                        iivf->own_invlists = true;
                        iivf->replace_invlists(odil, true);
                      }
                  }
                else
                  std::cerr << "cannot put index on disk : neither IVF nor "
                               "vectorTransform+IVF\n";
              }
          }
        _index_size = 0;
      }

#ifdef USE_GPU_FAISS
    if (_gpu)
      {
        if (_gpuids.size() == 0)
          {
            int ngpus = faiss::gpu::getNumDevices();
            for (int i = 0; i < ngpus; i++)
              _gpuids.push_back(i);
          }
        for (unsigned int i = 0; i < _gpuids.size(); ++i)
          {
            _gpu_res.push_back(new faiss::gpu::StandardGpuResources);
          }

        if (_gpuids.size() > 1)
          {
            faiss::Index *gindex = faiss::gpu::index_cpu_to_gpu_multiple(
                _gpu_res, _gpuids, _findex);
            delete _findex;
            _findex = gindex;
          }
        else
          {
            faiss::Index *gindex = faiss::gpu::index_cpu_to_gpu(
                _gpu_res[0], _gpuids[0], _findex);
            delete _findex;
            _findex = gindex;
          }
      }
#endif

    std::string db_filename = _model_repo + "/" + _db_name;
    if (fileops::file_exists(db_filename))
      {
        std::cerr << "open existing index db\n";
        _db->Open(db_filename, db::WRITE);
      }
    else
      {
        std::cerr << "create index db\n";
        _db->Open(db_filename, db::NEW);
      }
  }

  void FaissSE::train()
  {
    // train
    try
      {
        _findex->train(_train_samples.size() / _f, _train_samples.data());
      }
    catch (std::exception &e)
      {
        std::cerr << "could not train, maybe not enough data to train with "
                     "selected index type. index likely to be  empty"
                  << e.what() << std::endl;
        return;
      }
    // then add data to index
    _findex->add(_train_samples.size() / _f, _train_samples.data());
    // then throw away data
    _train_samples.clear();
  }

  void FaissSE::update_index()
  {
    if (!_findex->is_trained)
      train();
    std::string index_path = _model_repo + "/" + _index_name;
#ifdef USE_GPU_FAISS
    if (_gpu)
      {
        faiss::Index *cindex = faiss::gpu::index_gpu_to_cpu(_findex);
        faiss::write_index(cindex, index_path.c_str());
        delete cindex;
      }
    else
      {
        faiss::write_index(_findex, index_path.c_str());
      }
#else
    faiss::write_index(_findex, index_path.c_str());
#endif
    _txn->Commit();
    _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
  }

  void FaissSE::remove_index()
  {
    fileops::remove_file(_model_repo, _index_name);
    fileops::remove_file(_model_repo, _il_name);
    std::string db_filename = _model_repo + "/" + _db_name;
    fileops::clear_directory(db_filename);
    rmdir(db_filename.c_str());
  }

  void FaissSE::index(const URIData &uri, const std::vector<double> &data)
  {
    if (!_findex->is_trained && _index_size >= _train_samples_size)
      train();
    long int idx = _index_size;
    if (_findex->is_trained)
      {
        std::vector<float> d(data.begin(), data.end());
        _findex->add(1, d.data());
      }
    else
      {
        _train_samples.insert(_train_samples.end(), data.begin(), data.end());
      }
    ++_index_size;
    add_to_db(idx, uri);
  }

  void FaissSE::index(const std::vector<URIData> &uris,
                      const std::vector<std::vector<double>> &datas)
  {
    if (!_findex->is_trained && _index_size >= _train_samples_size)
      train();
    long int idx = _index_size;
    if (_findex->is_trained)
      {
        std::vector<float> d;
        for (std::vector<double> data : datas)
          d.insert(d.end(), data.begin(), data.end());
        _findex->add(uris.size(), d.data());
      }
    else
      {
        for (std::vector<double> data : datas)
          _train_samples.insert(_train_samples.end(), data.begin(),
                                data.end());
      }
    _index_size += uris.size();
    for (unsigned long int i = 0; i < uris.size(); ++i)
      add_to_db(idx + i, uris[i]);
  }

  void FaissSE::search(const std::vector<double> &vec, const int &nn,
                       std::vector<URIData> &uris,
                       std::vector<double> &distances)
  {
    if (!_findex->is_trained)
      train();
    std::vector<long int> labels(nn, -1);
    std::vector<float> d(nn, -1.0);
    std::vector<float> v(vec.begin(), vec.end());
    faiss::IndexIVF *iivf = dynamic_cast<faiss::IndexIVF *>(_findex);
    if (iivf)
      {
        if (_nprobe == -1)
          {
            int np = iivf->nlist / 50;
            if (np < 2)
              iivf->nprobe = 2;
            else
              iivf->nprobe = np;
          }
        else
          iivf->nprobe = _nprobe;
      }

    _findex->search(1, v.data(), nn, d.data(), labels.data());
    for (int i = 0; i < nn; ++i)
      {
        long int label = labels[i];
        if (label != -1)
          {
            URIData uri;
            get_from_db(label, uri);
            uris.push_back(uri);
            distances.push_back(d[i] / ((double)vec.size()));
          }
      }
  }

  void FaissSE::add_to_db(const int &idx, const URIData &fmap)
  {
    if (_count_put == 0)
      _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
    _txn->Put(std::to_string(idx), fmap.encode());
    ++_count_put;
    if (_count_put % _count_put_max == 0)
      {
        _txn->Commit(); // batch commit
        _txn = std::unique_ptr<db::Transaction>(_db->NewTransaction());
      }
  }

  void FaissSE::get_from_db(const int &idx, URIData &fmap)
  {
    std::string tmp;
    _db->Get(std::to_string(idx), tmp);
    fmap.decode(tmp);
  }

  template class SearchEngine<FaissSE>;

#endif
}
