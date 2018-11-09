/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#include "outputconnectorstrategy.h"
#include <thread>

// NCNN
#include "ncnnlib.h"
#include "ncnninputconns.h"
#include "cpu.h"
#include "net.h"

namespace dd
{
    int my_hardware_concurrency()
    {
        std::ifstream cpuinfo("/proc/cpuinfo");

        return std::count(std::istream_iterator<std::string>(cpuinfo),
			  std::istream_iterator<std::string>(),
			  std::string("processor"));
    }
  
    unsigned int hardware_concurrency()
    {
        unsigned int cores = std::thread::hardware_concurrency();
        if (!cores)
            cores = my_hardware_concurrency();
        if (cores > 4)
            cores /= 2;
        return cores;
    }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    ncnn::UnlockedPoolAllocator NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::_blob_pool_allocator
      = ncnn::UnlockedPoolAllocator();
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    ncnn::PoolAllocator NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::_workspace_pool_allocator
      = ncnn::PoolAllocator();
  
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::NCNNLib(const NCNNModel &cmodel)
        :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,NCNNModel>(cmodel)
    {
        this->_libname = "ncnn";
        _net = new ncnn::Net();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::NCNNLib(NCNNLib &&tl) noexcept
        :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,NCNNModel>(std::move(tl))
    {
        this->_libname = "ncnn";
	_net = tl._net;
	tl._net = nullptr;
	_nclasses = tl._nclasses;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~NCNNLib()
    {
      delete _net;
      _net = nullptr;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
    {
      /*ncnn::set_omp_dynamic(0);
	ncnn::set_omp_num_threads(hardware_concurrency());*/
      _net->load_param(this->_mlmodel._params.c_str());
      _net->load_model(this->_mlmodel._weights.c_str());
      if (ad.has("nclasses"))
	_nclasses = ad.get("nclasses").get<int>();
      /*_blob_pool_allocator.clear();
	_workspace_pool_allocator.clear();*/
      _blob_pool_allocator.set_size_compare_ratio(0.0f);
      _workspace_pool_allocator.set_size_compare_ratio(0.5f);
      ncnn::Option opt;
      opt.lightmode = true;
      opt.num_threads = hardware_concurrency();
      opt.blob_allocator = &_blob_pool_allocator;
      opt.workspace_allocator = &_workspace_pool_allocator;
      ncnn::set_default_option(opt);
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
    {
        (void)ad;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
                                        APIData &out)
    {
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										APIData &out)
    {
        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy tout;
        try {
            inputc.transform(ad);
        } catch (...) {
            throw;
        }

        ncnn::Extractor ex = _net->create_extractor();
        ex.set_num_threads(hardware_concurrency());
        ex.input("data", inputc._in);

        APIData ad_output = ad.getobj("parameters").getobj("output");

        // Get bbox
        bool bbox = false;
        if (ad_output.has("bbox") && ad_output.get("bbox").get<bool>())
            bbox = true;

        // Extract detection or classification
        if (bbox == true) {
            ex.extract("detection_out", inputc._out);
        } else {
            ex.extract("prob", inputc._out);
        }

        std::vector<APIData> vrad;
        std::vector<double> probs;
        std::vector<std::string> cats;
        std::vector<APIData> bboxes;
        APIData rad;
	
        // Get confidence_threshold
        double confidence_threshold = 0.0;
        if (ad_output.has("confidence_threshold")) {
	  try {
	    confidence_threshold = ad_output.get("confidence_threshold").get<double>();
	  } catch (std::exception &e) {
	    confidence_threshold = static_cast<double>(ad_output.get("confidence_threshold").get<int>());
	  }
        }

        // Get best
        int best = 1;
        if (ad_output.has("best")) {
            best = ad_output.get("best").get<int>();
        }

        if (bbox == true)
	  {
	    int rows = inputc.height();
	    int cols = inputc.width();
	    
            for (int i = 0; i < inputc._out.h; i++) {
                const float* values = inputc._out.row(i);
                if (values[1] < confidence_threshold)
                    continue;

                cats.push_back(this->_mlmodel.get_hcorresp(values[0]));
                probs.push_back(values[1]);

                APIData ad_bbox;
                ad_bbox.add("xmin",values[2] * cols);
                ad_bbox.add("ymax",values[3] * rows);
                ad_bbox.add("xmax",values[4] * cols);
                ad_bbox.add("ymin",values[5] * rows);
                bboxes.push_back(ad_bbox);
            }
        } else {
            std::vector<float> cls_scores;

            cls_scores.resize(inputc._out.w);
            for (int j = 0; j < inputc._out.w; j++) {
                cls_scores[j] = inputc._out[j];
            }
            int size = cls_scores.size();
            std::vector< std::pair<float, int> > vec;
            vec.resize(size);
            for (int i = 0; i < size; i++) {
                vec[i] = std::make_pair(cls_scores[i], i);
            }
        
            std::partial_sort(vec.begin(), vec.begin() + best, vec.end(),
                              std::greater< std::pair<float, int> >());
        
            for (int i = 0; i < best; i++)
            {
                if (vec[i].first < confidence_threshold)
                    continue;
                cats.push_back(this->_mlmodel.get_hcorresp(vec[i].second));
                probs.push_back(vec[i].first);
            }
        }

        rad.add("uri", "1");
        rad.add("loss", 0.0);
        rad.add("probs", probs);
        rad.add("cats", cats);
        if (bbox == true)
            rad.add("bboxes", bboxes);

        vrad.push_back(rad);
	tout.add_results(vrad);
        out.add("nclasses", this->_nclasses);
        if (bbox == true)
            out.add("bbox", true);
        out.add("roi", false);
        out.add("multibox_rois", false);
        tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));
        out.add("status", 0);
	return 0;
    }

    template class NCNNLib<ImgNCNNInputFileConn,SupervisedOutput,NCNNModel>;
}
