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

namespace dd
{

  /*-- URIData --*/
  char URIData::_enc_char = '^';
  
  std::string URIData::encode() const
  {
    if (_bbox.empty())
      return _uri;
    std::string enc = _uri + _enc_char;
    for (auto b: _bbox)
      enc += std::to_string(b) + _enc_char;
    enc += std::to_string(_prob) + _enc_char;
    enc += _cat;
    return enc;
  }

  void URIData::decode(const std::string &tmp)
  {
    std::vector<std::string> tok = dd_utils::split(tmp,_enc_char);
    _uri = tok.at(0);
    if (tok.size() == 1)
      return;
    for (int i=1;i<5;i++) // bbox is 4 double
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
    :_dim(dim)
  {
    _tse = new TSE(_dim,model_repo);
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
  void SearchEngine<TSE>::remove_index()
  {
    std::cerr << "removing index\n";
    _tse->remove_index();
  }
  
  template <class TSE>
  void SearchEngine<TSE>::index(const URIData &uri,
				const std::vector<double> &data)
  {
    std::lock_guard<std::mutex> lock(_index_mutex);
    _tse->index(uri,data);
  }
  
  template <class TSE>
  void SearchEngine<TSE>::search(const std::vector<double> &data,
				 const int &nn,
				 std::vector<URIData> &uris,
				 std::vector<double> &distances)
  {
    _tse->search(data,nn,uris,distances);
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
      {
	_saved_tree = true;
	_aindex->load(index_filename.c_str(),_map_populate);
	_built_index = true;
      }
    std::string db_filename = _model_repo + "/" + _db_name;
    if (fileops::file_exists(db_filename))
      {
	std::cerr << "open existing index db\n";
	_db->Open(db_filename,caffe::db::WRITE);
      }
    else
      {
	std::cerr << "create index db\n";
	_db->Open(db_filename,caffe::db::NEW);
      }
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
    if (_count_put % _count_put_max != 0)
      {
	_txn->Commit(); // last pending db commit
	_txn = std::unique_ptr<caffe::db::Transaction>(_db->NewTransaction());
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
    _aindex->save(index_path.c_str(),_map_populate);
    _saved_tree = true;
  }
  
  // must be protected by mutex
  void AnnoySE::index(const URIData &uri,
		      const std::vector<double> &vec)
  {
    if (_saved_tree)
      throw SimIndexException("Cannot index after Annoy index has been saved");
    int idx = _index_size;
    _aindex->add_item(idx,&vec[0]);
    ++_index_size;
    add_to_db(idx,uri);
  }
  
  void AnnoySE::search(const std::vector<double> &vec,
		       const int &nn,
		       std::vector<URIData> &uris,
		       std::vector<double> &distances)
  {
    if (!_built_index)
      throw SimSearchException("Cannot search before the Annoy tree has been built");
    std::vector<int> result;
    _aindex->get_nns_by_vector(&vec[0],nn,-1,&result,&distances);
    for (auto i: result)
      {
	URIData uri;
	get_from_db(i,uri);
	uris.push_back(uri);
      }
  }

  void AnnoySE::add_to_db(const int &idx,
			  const URIData &fmap)
  {
    if (_count_put == 0)
      _txn = std::unique_ptr<caffe::db::Transaction>(_db->NewTransaction());
    _txn->Put(std::to_string(idx),fmap.encode());
    ++_count_put;
    if (_count_put % _count_put_max == 0)
      {
	_txn->Commit(); // batch commit
	_txn = std::unique_ptr<caffe::db::Transaction>(_db->NewTransaction());
      }
  }
  
  void AnnoySE::get_from_db(const int &idx,
			    URIData &fmap)
  {
    std::string tmp;
    _db->Get(std::to_string(idx),tmp);
    fmap.decode(tmp);
  }
  

  template class SearchEngine<AnnoySE>;
}
