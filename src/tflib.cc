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

#include "tflib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TFLib(const TFModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TFModel>(cmodel)
  {
    this->_libname = "tensorflow";
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TFLib(TFLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TFModel>(std::move(cl))
  {
    this->_libname = "tensorflow";
    _nclasses = cl._nclasses;
    _regression = cl._regression;
    _ntargets = cl._ntargets;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~TFLib()
  {
    //TODO: delete structures, if any
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
	_regression = true;
	_nclasses = 1;
      }
    if (ad.has("ntargets"))
      _ntargets = ad.get("ntargets").get<int>();
    if (_nclasses == 0)
      throw MLLibBadParamException("number of classes is unknown (nclasses == 0)");
    if (_regression && _ntargets == 0)
      throw MLLibBadParamException("number of regression targets is unknown (ntargets == 0)");
    this->_mlmodel.read_from_repository();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    //TODO
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
									       APIData &out)
  {
    //TODO
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    //TODO
  }

  template class TFLib<ImgTFInputFileConn,SupervisedOutput,TFModel>;
  template class TFLib<ImgTFInputFileConn,UnsupervisedOutput,TFModel>;
  
}
