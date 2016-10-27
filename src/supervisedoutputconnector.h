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

#ifndef SUPERVISEDOUTPUTCONNECTOR_H
#define SUPERVISEDOUTPUTCONNECTOR_H

namespace dd
{

    /**
   * \brief supervised machine learning output connector class
   */
  class SupervisedOutput : public OutputConnectorStrategy
  {
  public:

    /**
     * \brief supervised result
     */
    class sup_result
    {
    public:
      /**
       * \brief constructor
       * @param loss result loss
       */
      sup_result(const std::string &label, const double &loss=0.0)
	:_label(label),_loss(loss) {}
      
      /**
       * \brief destructor
       */
      ~sup_result() {}

      /**
       * \brief add category to result
       * @param prob category predicted probability
       * @Param cat category name
       */
      inline void add_cat(const double &prob, const std::string &cat)
      {
	_cats.insert(std::pair<double,std::string>(prob,cat));
      }

      std::string _label;
      double _loss = 0.0; /**< result loss. */
      std::map<double,std::string,std::greater<double>> _cats; /**< categories and probabilities for this result */
    };

  public:
    /**
     * \brief supervised output connector constructor
     */
  SupervisedOutput()
      :OutputConnectorStrategy()
      {
      }
    
    /**
     * \brief supervised output connector copy-constructor
     * @param sout supervised output connector
     */
    SupervisedOutput(const SupervisedOutput &sout)
      :OutputConnectorStrategy(),_best(sout._best)
      {
      }
    
    ~SupervisedOutput() {}

    /**
     * \brief supervised output connector initialization
     * @param ad data object for "parameters/output"
     */
    void init(const APIData &ad)
    {
      APIData ad_out = ad.getobj("parameters").getobj("output");
      if (ad_out.has("best"))
	_best = ad_out.get("best").get<int>();
    }

    /**
     * \brief add prediction result to supervised connector output
     * @param vrad vector of data objects
     */
    inline void add_results(const std::vector<APIData> &vrad)
    {
      std::unordered_map<std::string,int>::iterator hit;
      for (APIData ad: vrad)
	{
	  std::string uri = ad.get("uri").get<std::string>();
	  double loss = ad.get("loss").get<double>();
	  std::vector<double> probs = ad.get("probs").get<std::vector<double>>();
	  std::vector<std::string> cats = ad.get("cats").get<std::vector<std::string>>();
	  if ((hit=_vcats.find(uri))==_vcats.end())
	    {
	      auto resit = _vcats.insert(std::pair<std::string,int>(uri,_vvcats.size()));
	      _vvcats.push_back(sup_result(uri,loss));
	      hit = resit.first;
	      for (size_t i=0;i<probs.size();i++)
		{
		  _vvcats.at((*hit).second).add_cat(probs.at(i),cats.at(i));
		}
	    }
	}
    }
    
    /**
     * \brief best categories selection from results
     * @param ad_out output data object
     * @param bcats supervised output connector
     */
    void best_cats(const APIData &ad_out, SupervisedOutput &bcats) const
    {
      int best = _best;
      if (ad_out.has("best"))
	best = ad_out.get("best").get<int>();
      for (size_t i=0;i<_vvcats.size();i++)
	{
	  sup_result sresult = _vvcats.at(i);
	  sup_result bsresult(sresult._label,sresult._loss);
	  std::copy_n(sresult._cats.begin(),std::min(best,static_cast<int>(sresult._cats.size())),
		      std::inserter(bsresult._cats,bsresult._cats.end()));
	  bcats._vcats.insert(std::pair<std::string,int>(sresult._label,bcats._vvcats.size()));
	  bcats._vvcats.push_back(bsresult);
	}
    }

    /**
     * \brief finalize output supervised connector data
     * @param ad_in data output object from the API call
     * @param ad_out data object as the call response
     */
    void finalize(const APIData &ad_in, APIData &ad_out)
    {
      SupervisedOutput bcats(*this);
      bool regression = false;
      bool autoencoder = false;
      if (ad_out.has("regression"))
	{
	  if (ad_out.get("regression").get<bool>())
	    {
	      regression = true;
	      _best = ad_out.get("nclasses").get<int>();
	    }
	  ad_out.erase("regression");
	  ad_out.erase("nclasses");
	}
      if (ad_out.has("autoencoder") && ad_out.get("autoencoder").get<bool>())
	{
	  autoencoder = true;
	  _best = 1;
	  ad_out.erase("autoencoder");
	}
      best_cats(ad_in,bcats);
      bcats.to_ad(ad_out,regression,autoencoder);
    }
    
    struct PredictionAndAnswer {
      float prediction;
      unsigned char answer; //this is either 0 or 1
    };

    // measure
    static void measure(const APIData &ad_res, const APIData &ad_out, APIData &out)
    {
      APIData meas_out;
      bool tloss = ad_res.has("train_loss");
      bool loss = ad_res.has("loss");
      bool iter = ad_res.has("iteration");
      bool regression = ad_res.has("regression");
      if (ad_out.has("measure"))
	{
	  std::vector<std::string> measures = ad_out.get("measure").get<std::vector<std::string>>();
      	  bool bauc = (std::find(measures.begin(),measures.end(),"auc")!=measures.end());
	  bool bacc = false;
	  for (auto s: measures)
	    if (s.find("acc")!=std::string::npos)
	      {
		bacc = true;
		break;
	      }
	  bool bf1 = (std::find(measures.begin(),measures.end(),"f1")!=measures.end());
	  bool bmcll = (std::find(measures.begin(),measures.end(),"mcll")!=measures.end());
	  bool bgini = (std::find(measures.begin(),measures.end(),"gini")!=measures.end());
	  bool beucll = (std::find(measures.begin(),measures.end(),"eucll")!=measures.end());
	  if (bauc) // XXX: applies two binary classification problems only
	    {
	      double mauc = auc(ad_res);
	      meas_out.add("auc",mauc);
	    }
	  if (bacc)
	    {
	      std::map<std::string,double> accs = acc(ad_res,measures);
	      auto mit = accs.begin();
	      while(mit!=accs.end())
		{
		  meas_out.add((*mit).first,(*mit).second);
		  ++mit;
		}
	    }
	  if (bf1)
	    {
	      double f1,precision,recall,acc;
	      dMat conf_diag,conf_matrix;
	      f1 = mf1(ad_res,precision,recall,acc,conf_diag,conf_matrix);
	      meas_out.add("f1",f1);
	      meas_out.add("precision",precision);
	      meas_out.add("recall",recall);
	      meas_out.add("accp",acc);
	      if (std::find(measures.begin(),measures.end(),"cmdiag")!=measures.end())
		{
		  std::vector<double> cmdiagv;
		  for (int i=0;i<conf_diag.rows();i++)
		    cmdiagv.push_back(conf_diag(i,0));
		  meas_out.add("cmdiag",cmdiagv);
		}
	      if (std::find(measures.begin(),measures.end(),"cmfull")!=measures.end())
		{
		  std::vector<std::string> clnames = ad_res.get("clnames").get<std::vector<std::string>>();
		  APIData cmobj;
		  cmobj.add("labels",clnames);
		  std::vector<APIData> cmdata;
		  for (int i=0;i<conf_matrix.cols();i++)
		    {
		      std::vector<double> cmrow;
		      for (int j=0;j<conf_matrix.rows();j++)
			cmrow.push_back(conf_matrix(j,i));
		      APIData adrow;
		      adrow.add(clnames.at(i),cmrow);
		      cmdata.push_back(adrow);
		    }
		  std::vector<APIData> vcmdata = {cmdata};
		  meas_out.add("cmfull",vcmdata);
		}
	    }
	  if (bmcll)
	    {
	      double mmcll = mcll(ad_res);
	      meas_out.add("mcll",mmcll);
	    }
	  if (bgini)
	    {
	      double mgini = gini(ad_res,regression);
	      meas_out.add("gini",mgini);
	    }
	  if (beucll)
	    {
	      double meucll = eucll(ad_res);
	      meas_out.add("eucll",meucll);
	    }
	}
	if (loss)
	  meas_out.add("loss",ad_res.get("loss").get<double>()); // 'universal', comes from algorithm
	if (tloss)
	  meas_out.add("train_loss",ad_res.get("train_loss").get<double>());
	if (iter)
	  meas_out.add("iteration",ad_res.get("iteration").get<double>());
	std::vector<APIData> vad = { meas_out };
	out.add("measure",vad);
    }

    // measure: ACC
    static std::map<std::string,double> acc(const APIData &ad,
					    const std::vector<std::string> &measures)
    {
      struct acc_comp
      {
        acc_comp(const std::vector<double> &v)
	:_v(v) {}
	bool operator()(double a, double b) { return _v[a] > _v[b]; }
	const std::vector<double> _v;
      };
      std::map<std::string,double> accs;
      std::vector<int> vacck;
      for(auto s: measures)
	if (s.find("acc")!=std::string::npos)
	  {
	    std::vector<std::string> sv = dd_utils::split(s,'-');
	    if (sv.size() == 2)
	      {
		vacck.push_back(std::atoi(sv.at(1).c_str()));
	      }
	    else vacck.push_back(1);
	  }
      
      int batch_size = ad.get("batch_size").get<int>();
      for (auto k: vacck)
	{
	  double acc = 0.0;
	  for (int i=0;i<batch_size;i++)
	    {
	      APIData bad = ad.getobj(std::to_string(i));
	      std::vector<double> predictions = bad.get("pred").get<std::vector<double>>();
	      if (k-1 >= static_cast<int>(predictions.size()))
		continue; // ignore instead of error
	      std::vector<int> predk(predictions.size());
	      for (size_t j=0;j<predictions.size();j++)
		predk[j] = j;
	      std::partial_sort(predk.begin(),predk.begin()+k-1,predk.end(),acc_comp(predictions));
	      for (int l=0;l<k;l++)
		if (predk.at(l) == bad.get("target").get<double>())
		  {
		    acc++;
		    break;
		  }
	    }
	  std::string key = "acc";
	  if (k>1)
	    key += "-" + std::to_string(k);
	  accs.insert(std::pair<std::string,double>(key,acc / static_cast<double>(batch_size)));
	}
      return accs;
    }

    // measure: F1
    static double mf1(const APIData &ad, double &precision, double &recall, double &acc, dMat &conf_diag, dMat &conf_matrix)
    {
      int nclasses = ad.get("nclasses").get<int>();
      double f1=0.0;
      conf_matrix = dMat::Zero(nclasses,nclasses);
      int batch_size = ad.get("batch_size").get<int>();
      for (int i=0;i<batch_size;i++)
	{
	  APIData bad = ad.getobj(std::to_string(i));
	  std::vector<double> predictions = bad.get("pred").get<std::vector<double>>();
	  int maxpr = std::distance(predictions.begin(),std::max_element(predictions.begin(),predictions.end()));
	  double target = bad.get("target").get<double>();
	  if (target < 0)
	    throw OutputConnectorBadParamException("negative supervised discrete target (e.g. wrong use of label_offset ?");
	  else if (target >= nclasses)
	    throw OutputConnectorBadParamException("target class has id " + std::to_string(target) + " is higher than the number of classes " + std::to_string(nclasses) + " (e.g. wrong number of classes specified with nclasses");
	  conf_matrix(maxpr,target) += 1.0;
	}
      conf_diag = conf_matrix.diagonal();
      dMat conf_csum = conf_matrix.colwise().sum();
      dMat conf_rsum = conf_matrix.rowwise().sum();
      dMat eps = dMat::Constant(nclasses,1,1e-8);
      acc = conf_diag.sum() / conf_matrix.sum();
      precision = conf_diag.transpose().cwiseQuotient(conf_csum + eps.transpose()).sum() / static_cast<double>(nclasses);
      recall = conf_diag.cwiseQuotient(conf_rsum + eps).sum() / static_cast<double>(nclasses);
      f1 = (2.0*precision*recall) / (precision+recall);
      conf_diag = conf_diag.transpose().cwiseQuotient(conf_csum+eps.transpose()).transpose();
      for (int i=0;i<conf_matrix.cols();i++)
	conf_matrix.col(i) /= conf_csum(i);
      return f1;
    }
    
    // measure: AUC
    static double auc(const APIData &ad)
    {
      std::vector<double> pred1;
      std::vector<double> targets;
      int batch_size = ad.get("batch_size").get<int>();
      for (int i=0;i<batch_size;i++)
	{
	  APIData bad = ad.getobj(std::to_string(i));
	  pred1.push_back(bad.get("pred").get<std::vector<double>>().at(1));
	  targets.push_back(bad.get("target").get<double>());
	}
      return auc(pred1,targets);
    }
    static double auc(const std::vector<double> &pred, const std::vector<double> &targets)
    {
      class PredictionAndAnswer {
      public:
	PredictionAndAnswer(const float &f, const int &i)
	  :prediction(f),answer(i) {}
	~PredictionAndAnswer() {}
	float prediction;
	int answer; //this is either 0 or 1
      };
      std::vector<PredictionAndAnswer> p;
      for (size_t i=0;i<pred.size();i++)
	p.emplace_back(pred.at(i),targets.at(i));
      int count = p.size();
      
      std::sort(p.begin(),p.end(),
		[](const PredictionAndAnswer &p1, const PredictionAndAnswer &p2){return p1.prediction < p2.prediction;});

      int i,truePos,tp0,accum,tn,ones=0;
      float threshold; //predictions <= threshold are classified as zeros

      for (i=0;i<count;i++) ones+=p[i].answer;
      if (0==ones || count==ones) return 1;

      truePos=tp0=ones; accum=tn=0; threshold=p[0].prediction;
      for (i=0;i<count;i++) {
        if (p[i].prediction!=threshold) { //threshold changes
	  threshold=p[i].prediction;
	  accum+=tn*(truePos+tp0); //2* the area of trapezoid
	  tp0=truePos;
	  tn=0;
        }
        tn+= 1- p[i].answer; //x-distance between adjacent points
        truePos-= p[i].answer;            
      }
      accum+=tn*(truePos+tp0); //2* the area of trapezoid
      return accum/static_cast<double>((2*ones*(count-ones)));
    }
    
    // measure: multiclass logarithmic loss
    static double mcll(const APIData &ad)
    {
      double ll=0.0;
      int batch_size = ad.get("batch_size").get<int>();
      for (int i=0;i<batch_size;i++)
	{
	  APIData bad = ad.getobj(std::to_string(i));
	  std::vector<double> predictions = bad.get("pred").get<std::vector<double>>();
	  double target = bad.get("target").get<double>();
	  ll -= std::log(predictions.at(target));
	}
      return ll / static_cast<double>(batch_size);
    }

    static double eucll(const APIData &ad)
    {
      double eucl = 0.0;
      int batch_size = ad.get("batch_size").get<int>();
      for (int i=0;i<batch_size;i++)
	{
	  APIData bad = ad.getobj(std::to_string(i));
	  std::vector<double> predictions = bad.get("pred").get<std::vector<double>>();
	  std::vector<double> target;
	  if (predictions.size() > 1)
	    target = bad.get("target").get<std::vector<double>>();
	  else target.push_back(bad.get("target").get<double>());
	  for (size_t i=0;i<target.size();i++)
	    eucl += (predictions.at(i)-target.at(i))*(predictions.at(i)-target.at(i));
	}
	return eucl / static_cast<double>(batch_size);
    }
    
    // measure: gini coefficient
    static double comp_gini(const std::vector<double> &a, const std::vector<double> &p) {
      struct K {double a, p;} k[a.size()];
      for (size_t i = 0; i != a.size(); ++i) k[i] = {a[i], p[i]};
      std::stable_sort(k, k+a.size(), [](const K &a, const K &b) {return a.p > b.p;});
      double accPopPercSum=0, accLossPercSum=0, giniSum=0, sum=0;
      for (auto &i: a) sum += i;
      for (auto &i: k) 
	{
	  accLossPercSum += i.a/sum;
	  accPopPercSum += 1.0/a.size();
	  giniSum += accLossPercSum-accPopPercSum;
	}
      return giniSum/a.size();
    }
    static double comp_gini_normalized(const std::vector<double> &a, const std::vector<double> &p) {
      return comp_gini(a, p)/comp_gini(a, a);
    }
    
    static double gini(const APIData &ad,
		       const bool &regression)
    {
      int batch_size = ad.get("batch_size").get<int>();
      std::vector<double> a(batch_size);
      std::vector<double> p(batch_size);
      for (int i=0;i<batch_size;i++)
	{
	  APIData bad = ad.getobj(std::to_string(i));
	  a.at(i) = bad.get("target").get<double>();
	  if (regression)
	    p.at(i) = bad.get("pred").get<std::vector<double>>().at(0); //XXX: could be vector for multi-dimensional regression -> TODO: in supervised mode, get best pred index ?
	  else
	    {
	      std::vector<double> allpreds = bad.get("pred").get<std::vector<double>>();
	      a.at(i) = std::distance(allpreds.begin(),std::max_element(allpreds.begin(),allpreds.end()));
	    }
	}
      return comp_gini_normalized(a,p);
    }
    
    // for debugging purposes.
    /**
     * \brief print supervised output to string
     */
    void to_str(std::string &out, const int &rmax) const
    {
      auto vit = _vcats.begin();
      while(vit!=_vcats.end())
	{
	  int count = 0;
	  out += "-------------\n";
	  out += (*vit).first + "\n";
	  auto mit = _vvcats.at((*vit).second)._cats.begin();
	  while(mit!=_vvcats.at((*vit).second)._cats.end()&&count<rmax)
	  {
	    out += "accuracy=" + std::to_string((*mit).first) + " -- cat=" + (*mit).second + "\n";
	    ++mit;
	    ++count;
	  }
	  ++vit;
	}
    }

    /**
     * \brief write supervised output object to data object
     * @param out data destination
     */
    void to_ad(APIData &out, const bool &regression, const bool &autoencoder) const
    {
      static std::string cl = "classes";
      static std::string ve = "vector";
      static std::string ae = "losses";
      static std::string phead = "prob";
      static std::string chead = "cat";
      static std::string vhead = "val";
      static std::string ahead = "loss";
      static std::string last = "last";
      std::vector<APIData> vpred;
      for (size_t i=0;i<_vvcats.size();i++)
	{
	  APIData adpred;
	  std::vector<APIData> v;
	  auto mit = _vvcats.at(i)._cats.begin();
	  while(mit!=_vvcats.at(i)._cats.end())
	    {
	      APIData nad;
	      if (!autoencoder)
		nad.add(chead,(*mit).second);
	      if (regression)
		nad.add(vhead,(*mit).first);
	      else if (autoencoder)
		nad.add(ahead,(*mit).first);
	      else nad.add(phead,(*mit).first);
	      ++mit;
	      if (mit == _vvcats.at(i)._cats.end())
		nad.add(last,true);
	      v.push_back(nad);
	    }
	  if (regression)
	    adpred.add(ve,v);
	  else if (autoencoder)
	    adpred.add(ae,v);
	  else adpred.add(cl,v);
	  if (_vvcats.at(i)._loss > 0.0) // XXX: not set by Caffe in prediction mode for now
	    adpred.add("loss",_vvcats.at(i)._loss);
	  adpred.add("uri",_vvcats.at(i)._label);
	  vpred.push_back(adpred);
	}
      out.add("predictions",vpred);
    }
    
    std::unordered_map<std::string,int> _vcats; /**< batch of results, per uri. */
    std::vector<sup_result> _vvcats; /**< ordered results, per uri. */
    
    // options
    int _best = 1;
  };
  
}

#endif
