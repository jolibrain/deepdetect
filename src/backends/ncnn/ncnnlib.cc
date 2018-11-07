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
        return cores;
    }
  
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::NCNNLib(const NCNNModel &cmodel)
        :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,NCNNModel>(cmodel)
    {
        this->_libname = "ncnn";
        _net = new ncnn::Net();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::NCNNLib(NCNNLib &&cl) noexcept
        :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,NCNNModel>(std::move(cl))
    {
        this->_libname = "ncnn";
        _net = cl._net;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~NCNNLib()
    {
    /*    if (_net)
            delete _net;*/
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
    {
        ncnn::set_omp_dynamic(0);
        ncnn::set_omp_num_threads(6);
        _net->load_param(this->_mlmodel._params.c_str());
        _net->load_model(this->_mlmodel._weights.c_str());
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
            //auto startTransform = std::chrono::system_clock::now();
            inputc.transform(ad);
            /* auto endTransform = std::chrono::system_clock::now();
            std::chrono::duration<double> timeTransform = endTransform - startTransform;
            std::cerr << "Transform duration: " << timeTransform.count() << "s" << std::endl; */
        } catch (...) {
            throw;
        }

        //auto startExtraction = std::chrono::system_clock::now();
        ncnn::Extractor ex = _net->create_extractor();
        ex.set_num_threads(6);
        ex.input("data", inputc._in);
        ex.extract("detection_out", inputc._out);
        /*auto endExtraction = std::chrono::system_clock::now();
        std::chrono::duration<double> timeExtraction = endExtraction - startExtraction;
        std::cerr << "Extract duration: " << timeExtraction.count() << "s" << std::endl; */

        std::vector<APIData> vrad;

        std::vector<double> probs;
        std::vector<std::string> cats;
        std::vector<APIData> bboxes;
        int cols;
        int rows;
        int _nclasses = -1;

        APIData rad;
                                                                                                                                                                           
        rows = inputc.height();
        cols = inputc.width();

        double confidence_threshold = 0.0;
        APIData ad_output = ad.getobj("parameters").getobj("output");
        if (ad_output.has("confidence_threshold")) {
	        try {
	            confidence_threshold = ad_output.get("confidence_threshold").get<double>();
	        } catch (std::exception &e) {
	            // Try from int
	            confidence_threshold = static_cast<double>(ad_output.get("confidence_threshold").get<int>());
	        }
        }

        bool bbox = false;
        if (ad_output.has("bbox") && ad_output.get("bbox").get<bool>())
            bbox = true;

        if (ad.has("nclasses"))
            _nclasses = ad.get("nclasses").get<int>();

        for (int i = 0; i < inputc._out.h; i++) {
            const float* values = inputc._out.row(i);
            if (values[1] < confidence_threshold)
                continue;

            cats.push_back(this->_mlmodel.get_hcorresp(values[0]));
            probs.push_back(values[1]);

            if (bbox == true) {
                APIData ad_bbox;
                ad_bbox.add("xmin",values[2] * cols);
                ad_bbox.add("ymax",values[3] * rows);
                ad_bbox.add("xmax",values[4] * cols);
                ad_bbox.add("ymin",values[5] * rows);
                bboxes.push_back(ad_bbox);
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

        out.add("nclasses", _nclasses);
        if (bbox == true)
            out.add("bbox", true);
        out.add("roi", false);
        out.add("multibox_rois", false);

        tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));

        out.add("status", 0);
    }

    template class NCNNLib<ImgNCNNInputFileConn,SupervisedOutput,NCNNModel>;
}
