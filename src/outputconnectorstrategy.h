/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#ifndef OUTPUTCONNECTORSTRATEGY_H
#define OUTPUTCONNECTORSTRATEGY_H

#include <map>
#include <iostream>

namespace dd
{

  /**
   * \brief main output connector class
   */
  class OutputConnectorStrategy
  {
  public:
    OutputConnectorStrategy() {}
    ~OutputConnectorStrategy() {}

    /**
     * \brief output data reading
     */
    //int transform() { return 1; }
    
    /**
     * \brief initialization of output data connector
     * @param ad data object for "parameters/output"
     */
    void init(const APIData &ad);

    //TODO: output templating.
  };

  /**
   * \brief no output connector class
   */
  class NoOutputConn : public OutputConnectorStrategy
  {
  public:
    NoOutputConn()
      :OutputConnectorStrategy() {}
    ~NoOutputConn() {}
  };

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
      sup_result(const double &loss=0.0)
	:_loss(loss) {}
      
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
     * @param uri result uri
     * @param loss result loss
     */
    inline void add_result(const std::string &uri, const double &loss)
    {
      std::unordered_map<std::string,sup_result>::iterator hit;
      if ((hit=_vcats.find(uri))==_vcats.end())
	_vcats.insert(std::pair<std::string,sup_result>(uri,sup_result(loss)));
      else (*hit).second._loss = loss;
    }

    /**
     * \brief add predicted category and probability to existing result
     * @param uri result uri
     * @param prob  category probability
     * @param cat category name
     */
    inline void add_cat(const std::string &uri, const double &prob, const std::string &cat)
    {
      std::unordered_map<std::string,sup_result>::iterator hit;
      if ((hit=_vcats.find(uri))!=_vcats.end())
	(*hit).second.add_cat(prob,cat);
      // XXX: else error ?
    }

    /**
     * \brief best categories selection from results
     * @param ad_out output data object
     * @param bcats supervised output connector
     */
    void best_cats(const APIData &ad_out, SupervisedOutput &bcats) const
    {
      int best = _best;
      //APIData ad_out = ad.getobj("parameters").getobj("output");
      if (ad_out.has("best"))
	best = ad_out.get("best").get<int>();
      auto hit = _vcats.begin();
      while(hit!=_vcats.end())
	{
	  sup_result bsresult((*hit).second._loss);
	  std::copy_n((*hit).second._cats.begin(),std::min(best,static_cast<int>((*hit).second._cats.size())),
		      std::inserter(bsresult._cats,bsresult._cats.end()));
	  bcats._vcats.insert(std::pair<std::string,sup_result>((*hit).first,bsresult));
	  ++hit;
	}
    }

    struct PredictionAndAnswer {
      float prediction;
      unsigned char answer; //this is either 0 or 1
    };

    //On input, p[] should be in ascending order by prediction, and _count^2
    //must be less than 2^33 (int is 32-bit so it won't overflow).
    //double auc(const PredictionAndAnswer*p, unsigned int _count)
    double auc(const std::vector<double> &pred, const std::vector<int> &targets)
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
      return (double)accum/(2*ones*(count-ones));
    }
    
    // for debugging purposes.
    /**
     * \brief print supervised output to string
     */
    void to_str(std::string &out) const
    {
      auto vit = _vcats.begin();
      while(vit!=_vcats.end())
	{
	  out += "-------------\n";
	  out += (*vit).first + "\n";
	  auto mit = (*vit).second._cats.begin();
	  while(mit!=(*vit).second._cats.end())
	  {
	    out += "accuracy=" + std::to_string((*mit).first) + " -- cat=" + (*mit).second + "\n";
	    ++mit;
	  }
	  ++vit;
	}
    }

    /**
     * \brief write supervised output object to data object
     * @param out data destination
     */
    void to_ad(APIData &out) const
    {
      static std::string cl = "classes";
      static std::string phead = "prob";
      static std::string chead = "cat";
      std::vector<APIData> vpred;
      
      //debug
      /*std::string str;
      to_str(str);
      std::cout << "witness=\n" << str << std::endl;*/
      //debug
      
      auto hit = _vcats.begin();
      while(hit!=_vcats.end())
	{
	  int i = 0;
	  std::vector<APIData> v;
	  auto mit = (*hit).second._cats.begin();
	  while(mit!=(*hit).second._cats.end())
	    {
	      APIData nad;
	      nad.add(chead,(*mit).second);
	      nad.add(phead,(*mit).first);
	      v.push_back(nad);
	      ++i;
	      ++mit;
	    }
	  APIData adpred;
	  adpred.add(cl,v);
	  adpred.add("loss",(*hit).second._loss);
	  adpred.add("uri",(*hit).first);
	  vpred.push_back(adpred);
	  ++hit;
	}
      out.add("predictions",vpred);
    }
    
    std::unordered_map<std::string,sup_result> _vcats; /**< batch of results, per uri. */
    
    // options
    int _best = 1;
  };
  
}

#endif
