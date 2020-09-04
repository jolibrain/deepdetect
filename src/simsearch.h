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

#ifndef SIMSEARCH_H
#define SIMSEARCH_H

#include "apidata.h"
#ifdef USE_ANNOY
#include "annoylib.h"
#include "kissrandom.h"
#else
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/AutoTune.h>
#ifdef USE_GPU_FAISS
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#endif
#endif
#include "caffe/util/db.hpp"
#include <mutex>

namespace dd
{
  /**
   * \brief Similarity search indexing exception
   */
  class SimIndexException : public std::exception
  {
  public:
    SimIndexException(const std::string &s) : _s(s)
    {
    }
    ~SimIndexException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief Similarity search searching exception
   */
  class SimSearchException : public std::exception
  {
  public:
    SimSearchException(const std::string &s) : _s(s)
    {
    }
    ~SimSearchException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief stored feature map
   */
  class URIData
  {
  public:
    URIData()
    {
    }
    URIData(const std::string &uri, const std::vector<double> &bbox = {},
            const double &prob = 0.0, const std::string &cat = "")
        : _uri(uri), _bbox(bbox), _prob(prob), _cat(cat)
    {
    }
    ~URIData()
    {
    }

    std::string encode() const;
    void decode(const std::string &str);

    std::string _uri;
    std::vector<double> _bbox;
    double _prob = 0.0;
    std::string _cat;
    static char _enc_char;
  };

  template <class TSE> class SearchEngine
  {
  public:
    SearchEngine(const int &dim, const std::string &model_repo);
    ~SearchEngine();

    void create_index();

    void update_index();

    void remove_index();

    void index(const URIData &uri, const std::vector<double> &data);

    // batch index
    void index(const std::vector<URIData> &uris,
               const std::vector<std::vector<double>> &data);

    void search(const std::vector<double> &data, const int &nn,
                std::vector<URIData> &uris, std::vector<double> &distances);

    const int _dim = 128; /**< indexed vector length. */
    TSE *_tse = nullptr;
    std::mutex _index_mutex; /**< mutex around indexing calls. */
  };

#ifdef USE_ANNOY
  class AnnoySE
  {
  public:
    AnnoySE(const int &f, const std::string &model_repo);
    ~AnnoySE();

    // interface
    void create_index();

    void update_index();

    void remove_index();

    void index(const URIData &uri, const std::vector<double> &data);

    void index(const std::vector<URIData> &uris,
               const std::vector<std::vector<double>> &datas);

    void search(const std::vector<double> &vec, const int &nn,
                std::vector<URIData> &uris, std::vector<double> &distances);

    // internal functions
    void build_tree();

    void unbuild_tree();

    void save_tree();

    void add_to_db(const int &idx, const URIData &fmap);

    void get_from_db(const int &idx, URIData &fmap);

    void set_ntrees(const int &ntrees)
    {
      _ntrees = ntrees;
    }

    int _f = 128;      /**< indexed vector length. */
    int _ntrees = 100; /**< number of trees. */
    AnnoyIndex<int, double, Angular, Kiss32Random> *_aindex = nullptr;
    int _index_size = 0;
    std::string _model_repo; /**< model directory */
    const std::string _db_name = "names.bin";
    const std::string _db_backend = "lmdb";
    caffe::db::DB *_db = nullptr;
    std::unique_ptr<caffe::db::Transaction> _txn;
    int _count_put = 0;
    int _count_put_max = 1000;
    const std::string _index_name = "index.ann";
    bool _saved_tree = false;  /**< whether the tree has been saved. */
    bool _built_index = false; /**< whether the index has been built. */
    bool _map_populate = true; /**< whether to use MAP_POPULATE when mmapping
                                  the full index. */
  };
#endif

#ifdef USE_FAISS
  class FaissSE
  {
  public:
    FaissSE(const int &f, const std::string &model_repo);
    ~FaissSE();

    // interface
    void create_index();

    void update_index();

    void remove_index();

    void index(const URIData &uri, const std::vector<double> &data);

    void index(const std::vector<URIData> &uris,
               const std::vector<std::vector<double>> &datas);

    void search(const std::vector<double> &vec, const int &nn,
                std::vector<URIData> &uris, std::vector<double> &distances);

    void train();
    void add_to_db(const int &idx, const URIData &fmap);
    void get_from_db(const int &idx, URIData &fmap);

    faiss::Index *_findex = nullptr;
    std::string _index_key;

    int _f = 128; /**< indexed vector length. */
    long int _index_size = 0;
    std::string _model_repo; /**< model directory */
    const std::string _db_name = "names.bin";
    const std::string _db_backend = "lmdb";
    const std::string _index_name = "index.faiss";
    const std::string _il_name = "index_mmap.faiss";
    caffe::db::DB *_db = nullptr;
    std::unique_ptr<caffe::db::Transaction> _txn;
    int _count_put = 0;
    int _count_put_max = 1000;
    int _train_samples_size = 100000;
    bool _ondisk = true;
    int _nprobe = -1;
    std::vector<float> _train_samples;

#ifdef USE_GPU_FAISS
    bool _gpu = false;
    faiss::Index *_gpu_index;
    std::vector<faiss::gpu::GpuResources *> _gpu_res;
    std::vector<int> _gpuids;
#endif
  };
#endif

}

#endif
