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

namespace dd
{

  template <class TSE>
  SearchEngine<TSE>::SearchEngine(const int &dim)
    :_dim(dim)
  {
    _tse = new TSE(_dim);
  }

  template <class TSE>
  SearchEngine<TSE>::~SearchEngine()
  {
    delete _tse;
  }
  
  template <class TSE>
  void SearchEngine<TSE>::create_index()
  {
    _tse->create_index();
  }

  template <class TSE>
  void SearchEngine<TSE>::update_index()
  {
    _tse->update_index();
  }

  template <class TSE>
  void SearchEngine<TSE>::index(const std::string &uri,
				const std::vector<double> &data)
  {
    std::lock_guard<std::mutex> lock(_index_mutex);
    _tse.index(uri,data);
  }
  
  template <class TSE>
  void SearchEngine<TSE>::search(const std::vector<double> &data,
				 const int &nn,
				 const std::vector<std::string> &uris,
				 const std::vector<double> &distances)
  {
    _tse.search(data,nn,uris,distances);
  }

  /*- AnnoySE -*/

  AnnoySE::AnnoySE(const int &f,
		   const std::string &model_repo)
    :_f(f),_model_repo(model_repo)
  {
    _aindex = new AnnoyIndex<int,double,Angular,Kiss32Random>(f);
    _db = caffe::db::GetDB(_db_backend);
  }

  AnnoySE::~AnnoySE()
  {
    delete _aindex;
    if (_db)
      _db->Close();
    delete _db;
  }

  void AnnoySE::create_index() //TODO: exception
  {
    std::string index_filename = _model_repo + "/" + _index_name;
    if (fileops::file_exists(index_filename))
      _aindex->load(index_filename.c_str());
    std::string db_filename = _model_repo + "/" + _db_name;
    if (fileops::file_exists(db_filename))
      _db->Open(db_filename,caffe::db::READ);
    else _db->Open(db_filename,caffe::db::NEW);
  }

  void AnnoySE::remove_index()
  {
    fileops::remove_file(_model_repo,_index_name);
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
    _aindex->build(_ntrees);
  }

  void AnnoySE::unbuild_tree()
  {
    _aindex->unbuild();
  }

  void AnnoySE::save_tree()
  {
    std::string index_path = _model_repo + "/" + _index_name;
    _aindex->save(index_path.c_str());
  }
  
  // must be protected by mutex
  void AnnoySE::index(const std::string &uri,
		      const std::vector<double> &vec)
  {
    int idx = _index_size;
    _aindex->add_item(idx,&vec[0]);
    ++_index_size;
    add_to_db(idx,uri);
  }

  void AnnoySE::search(const std::vector<double> &vec,
		       const int &nn,
		       std::vector<std::string> &uris,
		       std::vector<double> &distances)
  {
    std::vector<int> result;
    _aindex->get_nns_by_vector(&vec[0],nn,-1,&result,&distances);
    for (auto i: result)
      {
	std::string uri;
	get_from_db(i,uri);
	uris.push_back(uri);
      }
  }

  void AnnoySE::add_to_db(const int &idx,
			  const std::string &uri)
  {
    std::unique_ptr<caffe::db::Transaction> txn(_db->NewTransaction());
    txn->Put(std::to_string(idx),uri);
    txn->Commit(); // no batch commits at the moment
  }

  void AnnoySE::get_from_db(const int &idx,
			    std::string &uri)
  {
    _db->Get(std::to_string(idx),uri);
  }
  
}
