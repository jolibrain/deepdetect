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
#include "annoylib.h"
#include "kissrandom.h"
#include "caffe/util/db.hpp"
#include <mutex>

namespace dd
{
  template <class TSE>
    class SearchEngine
    {
    public:
      SearchEngine(const int &dim);
      ~SearchEngine();
      
      void create_index();

      void update_index();

      void index(const std::string &uri,
		 const std::vector<double> &data);
      
      void search(const std::vector<double> &data,
		  const int &nn,
		  const std::vector<std::string> &uris,
		  const std::vector<double> &distances);

      const int _dim = 128; /**< indexed vector length. */
      TSE *_tse = nullptr;
      std::mutex _index_mutex; /**< mutex around indexing calls. */
    };

  class AnnoySE
  {
  public:
    AnnoySE(const int &f, const std::string &model_repo);
    ~AnnoySE();

    // interface
    void create_index();

    void update_index();
    
    void remove_index();

    void index(const std::string &uri,
		 const std::vector<double> &data);
    
    void search(const std::vector<double> &vec,
		const int &nn,
		std::vector<std::string> &uris,
		std::vector<double> &distances);
      
    // internal functions
    void build_tree();

    void unbuild_tree();

    void save_tree();
    
    void add_to_db(const int &idx,
		   const std::string &uri);

    void get_from_db(const int &idx,
		     std::string &uri);

    void set_ntrees(const int &ntrees) { _ntrees = ntrees; }
    
    int _f = 128; /**< indexed vector length. */
    int _ntrees = 100; /**< number of trees. */
    AnnoyIndex<int,double,Angular,Kiss32Random> *_aindex = nullptr;
    int _index_size = 0;
    std::string _model_repo; /**< model directory */
    const std::string _db_name = "names.bin";
    const std::string _db_backend = "lmdb";
    caffe::db::DB *_db = nullptr;
    const std::string _index_name = "index.ann";
  };
  
}

#endif
