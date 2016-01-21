/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#ifndef XGBINPUTCONNS_H
#define XGBINPUTCONNS_H

#include "csvinputfileconn.h"
#include <data/parser.h> // dmlc
#include <xgboost/data.h>

//namespace dd
//{
  /**
   * \brief high-level data structure shared among XGBoost compatible connectors of DeepDetect
   */
  /*class XGBInputInterface
  {
  public:
    XGBInputInterface() {}
    XGBInputInterface(const XGBInputInterface &xii)
      {
      }
    //TODO
    };*/

  //TODO: straight CSV reader from XGBoost

namespace dmlc {
  namespace data {
    /**
     * \brief Parser to pass data from deepdetect's CSV input connector to XGBoost
     */
    template <typename IndexType>
      class DDCSVParser : public ParserImpl<IndexType>
      {
      public:
	explicit DDCSVParser()
	  :ParserImpl<IndexType>()
	  {}
	~DDCSVParser() {}

	virtual void BeforeFirst() {}
	virtual size_t BytesRead() const {}
	
	virtual bool ParseNext(std::vector<RowBlockContainer<IndexType>> *data)
	{
	  //TODO: test + this is for training set, need flag for testing/eval set as well
	  auto hit = _csvc->_csvdata.begin();
	  while(hit!=_csvc->_csvdata.end())
	    {
	      RowBlockContainer<IndexType> rbc;
	      auto lit = _csvc->_columns.begin();
	      for (int i=0;i<(int)(*hit)._v.size();i++)
		{
		  if (i == _csvc->_label_pos[0]) //TODO: multilabel ?
		    {
		      rbc.label.push_back((*hit)._v.at(i)+_csvc->_label_offset[0]);
		    }
		  else if (i == _csvc->_id_pos)
		    {
		      ++lit;
		      continue;
		    }
		  else if (std::find(_csvc->_label_pos.begin(),_csvc->_label_pos.end(),i)==_csvc->_label_pos.end())
		    {
		      rbc.index.push_back(static_cast<IndexType>(i));
		      rbc.value.push_back((*hit)._v.at(i));
		    }
		}
	      data->push_back(rbc);
	      ++hit;
	    }
	  return true;
	}

	void set_csvc(dd::CSVInputFileConn *csvc)
	{
	  _csvc = csvc;
	}

	std::vector<RowBlockContainer<IndexType>>* get_data() { return this->data_; }
	
      protected:
	dd::CSVInputFileConn *_csvc = nullptr;
      };
    
    template<typename IndexType>
      Parser<IndexType>* CreateDDCSVParser(const char *uri_,
					   unsigned part_index,
					   unsigned num_parts)
      {
	return new DDCSVParser<IndexType>();
      }
  }
  //DMLC_REGISTRY_ENABLE(ParserFactoryReg<uint32_t>);
}

namespace dd
{
  class CSVXGBInputFileConn : public CSVInputFileConn
  {
  public:
    CSVXGBInputFileConn()
      :CSVInputFileConn() {}
    CSVXGBInputFileConn(const CSVXGBInputFileConn &i)
      :CSVInputFileConn(i) {}
    ~CSVXGBInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }

    void transform(const APIData &ad)
    {
      try
	{
	  CSVInputFileConn::transform(ad);
	}
      catch (std::exception &e)
	{
	  throw;
	}
      
      //TODO
      if (!_xgb_csv_parser)
	_xgb_csv_parser = static_cast<dmlc::data::DDCSVParser<uint32_t>*>(dmlc::Parser<uint32_t>::Create("",0,0,"ddcsv"));
      _xgb_csv_parser->set_csvc(this);

      if (_m)
	delete _m;
      _m = xgboost::DMatrix::Create(_xgb_csv_parser);
    }
    
    dmlc::data::DDCSVParser<uint32_t> *_xgb_csv_parser = nullptr;
    xgboost::DMatrix *_m;
  };
  
}

#endif
