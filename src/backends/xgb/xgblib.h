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

#ifndef XGBLIB_H
#define XGBLIB_H

#include "mllibstrategy.h"
#include "xgbmodel.h"
#include <dmlc/build_config.h>
#include <xgboost/learner.h>

namespace xgboost
{
  struct CLIParam : public dmlc::Parameter<CLIParam> {
    /*! \brief whether silent */
    int silent = 0;
    /*! \brief whether evaluate training statistics */
    bool eval_train = false;
    /*! \brief number of boosting iterations */
    int num_round = 100;
    /*! \brief the period to save the model, 0 means only save the final round model */
    int save_period;
    /*! \brief the path of test model file, or file to restart training */
    std::string model_in = "NULL";
    /*! \brief the path of final model file, to be saved */
    std::string model_out = "NULL";
    /*! \brief name of predict file */
    std::string name_pred;
    /*!\brief limit number of trees in prediction */
    int ntree_limit = 0;
    /*!\brief whether to directly output margin value */
    bool pred_margin = false;
    /*! \brief the paths of validation data sets */
    std::vector<std::string> eval_data_paths;
    /*! \brief the names of the evaluation data used in output log */
    std::vector<std::string> eval_data_names;
    /*! \brief all the configurations */
    std::vector<std::pair<std::string, std::string> > cfg;
  };
}

template<typename T>
void add_to_cfg(const std::string &key,
		const T &val,
		std::vector<std::pair<std::string,std::string>> &cfg)
{
  cfg.push_back(std::make_pair(key,std::to_string(val)));
}
template<>
inline void add_to_cfg(const std::string &key,
		       const std::string &val,
		       std::vector<std::pair<std::string,std::string>> &cfg)
{
  cfg.push_back(std::make_pair(key,val));
}

namespace dd
{
  /**
   * XGBoost library wrapper for deepdetect
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=XGBModel>
    class XGBLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
    XGBLib(const XGBModel &xmodel);
    XGBLib(XGBLib &&xl) noexcept;
    ~XGBLib();

    /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &d);

    int train(const APIData &ad, APIData &out);

    int predict(const APIData &ad, APIData &out);

    /*- local functions -*/
    void test(const APIData &ad,
	      std::unique_ptr<xgboost::Learner> &learner,
	      xgboost::DMatrix *dtest,
	      APIData &out);
    
    template<typename T>
    void add_cfg_param(const std::string &key,
		       const T &val)
    {
      add_to_cfg(key,val,_params.cfg);
    }

    public:
    // general parameters
    int _nclasses = 0; /**< required. */
    bool _regression = false; /**< whether the net acts as a regressor. */
    int _ntargets = 0; /**< number of classification or regression targets. */
    std::string _booster = "gbtree"; /**< xgb booster, optional. */
    std::string _objective = "multi:softprob"; /**< xgb service objective. */

    bool _gpu = false; /**< whether to use GPU. */
    xgboost::CLIParam _params;
    xgboost::Learner *_learner = nullptr; /**< learner for prediction. */
    std::mutex _learner_mutex; /**< mutex around the learner, e.g.. no concurrent predict calls as learner is not re-instantiated. Use batches instead. */
    };

}

#endif
