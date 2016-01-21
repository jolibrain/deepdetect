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
#include <xgboost/learner.h>

namespace xgboost
{
  enum CLITask {
    kTrain = 0,
    kDump2Text = 1,
    kPredict = 2
  };
  
  struct CLIParam : public dmlc::Parameter<CLIParam> {
    /*! \brief the task name */
    int task;
    /*! \brief whether silent */
    int silent;
    /*! \brief whether evaluate training statistics */
    bool eval_train;
    /*! \brief number of boosting iterations */
    int num_round;
    /*! \brief the period to save the model, 0 means only save the final round model */
    int save_period;
    /*! \brief the path of training set */
    std::string train_path;
    /*! \brief path of test dataset */
    std::string test_path;
    /*! \brief the path of test model file, or file to restart training */
    std::string model_in;
    /*! \brief the path of final model file, to be saved */
    std::string model_out;
    /*! \brief the path of directory containing the saved models */
    std::string model_dir;
    /*! \brief name of predict file */
    std::string name_pred;
    /*! \brief data split mode */
    int dsplit;
    /*!\brief limit number of trees in prediction */
    int ntree_limit;
    /*!\brief whether to directly output margin value */
    bool pred_margin;
    /*! \brief whether dump statistics along with model */
    int dump_stats;
    /*! \brief name of feature map */
    std::string name_fmap;
    /*! \brief name of dump file */
    std::string name_dump;
    /*! \brief the paths of validation data sets */
    std::vector<std::string> eval_data_paths;
    /*! \brief the names of the evaluation data used in output log */
    std::vector<std::string> eval_data_names;
    /*! \brief all the configurations */
    std::vector<std::pair<std::string, std::string> > cfg;
    
    // declare parameters
    DMLC_DECLARE_PARAMETER(CLIParam) {
      // NOTE: declare everything except eval_data_paths.
      DMLC_DECLARE_FIELD(task).set_default(kTrain)
	.add_enum("train", kTrain)
	.add_enum("dump", kDump2Text)
	.add_enum("pred", kPredict)
	.describe("Task to be performed by the CLI program.");
      DMLC_DECLARE_FIELD(silent).set_default(0).set_range(0, 2)
	.describe("Silent level during the task.");
      DMLC_DECLARE_FIELD(eval_train).set_default(false)
	.describe("Whether evaluate on training data during training.");
      DMLC_DECLARE_FIELD(num_round).set_default(10).set_lower_bound(1)
	.describe("Number of boosting iterations");
      DMLC_DECLARE_FIELD(save_period).set_default(0).set_lower_bound(0)
	.describe("The period to save the model, 0 means only save final model.");
      DMLC_DECLARE_FIELD(train_path).set_default("NULL")
	.describe("Training data path.");
      DMLC_DECLARE_FIELD(test_path).set_default("NULL")
	.describe("Test data path.");
      DMLC_DECLARE_FIELD(model_in).set_default("NULL")
	.describe("Input model path, if any.");
      DMLC_DECLARE_FIELD(model_out).set_default("NULL")
	.describe("Output model path, if any.");
      DMLC_DECLARE_FIELD(model_dir).set_default("./")
	.describe("Output directory of period checkpoint.");
      DMLC_DECLARE_FIELD(name_pred).set_default("pred.txt")
	.describe("Name of the prediction file.");
      DMLC_DECLARE_FIELD(dsplit).set_default(0)
	.add_enum("auto", 0)
	.add_enum("col", 1)
	.add_enum("row", 2)
	.describe("Data split mode.");
      DMLC_DECLARE_FIELD(ntree_limit).set_default(0).set_lower_bound(0)
	.describe("Number of trees used for prediction, 0 means use all trees.");
      DMLC_DECLARE_FIELD(pred_margin).set_default(false)
	.describe("Whether to predict margin value instead of probability.");
      DMLC_DECLARE_FIELD(dump_stats).set_default(false)
	.describe("Whether dump the model statistics.");
      DMLC_DECLARE_FIELD(name_fmap).set_default("NULL")
	.describe("Name of the feature map file.");
      DMLC_DECLARE_FIELD(name_dump).set_default("dump.txt")
	.describe("Name of the output dump text file.");
      // alias
      DMLC_DECLARE_ALIAS(train_path, data);
      DMLC_DECLARE_ALIAS(test_path, test:data);
      DMLC_DECLARE_ALIAS(name_fmap, fmap);
    }
    
    // customized configure function of CLIParam
    /*inline void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) {
      this->cfg = cfg;
      this->InitAllowUnknown(cfg);
      for (const auto& kv : cfg) {
	if (!strncmp("eval[", kv.first.c_str(), 5)) {
	  char evname[256];
	  CHECK_EQ(sscanf(kv.first.c_str(), "eval[%[^]]", evname), 1)
	    << "must specify evaluation name for display";
	  eval_data_names.push_back(std::string(evname));
	  eval_data_paths.push_back(kv.second);
	}
      }
      // constraint.
      if (name_pred == "stdout") {
	save_period = 0;
	silent = 1;
      }
      if (dsplit == 0 && rabit::IsDistributed()) {
	dsplit = 2;
      }
      if (rabit::GetRank() != 0) {
	silent = 2;
      }
      }*/
  };
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
    //TODO: test() ...

    public:
    //xgboost::Learner *_xgblearn = nullptr;
    xgboost::CLIParam _params; //TODO: redo parameter structure
    };
  
}

#endif
