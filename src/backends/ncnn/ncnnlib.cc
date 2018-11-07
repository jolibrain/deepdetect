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
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::NCNNLib(NCNNLib &&cl) noexcept
        :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,NCNNModel>(std::move(cl))
    {
        this->_libname = "ncnn";
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~NCNNLib()
    {
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void NCNNLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
    {
        _net.load_param(this->_mlmodel._params.c_str());
        _net.load_model(this->_mlmodel._weights.c_str());
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
        std::cerr << "predict start" << std::endl;
        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy tout;
        try
        {
            inputc.transform(ad);
        }
        catch(...)
        {
            throw;
        }

        std::cerr << "after transform" << std::endl;
        ncnn::Net nnet;
        nnet.load_param(this->_mlmodel._params.c_str());
        nnet.load_model(this->_mlmodel._weights.c_str());
        ncnn::Extractor ex = nnet.create_extractor();
        std::cerr << "extractor created" << std::endl;
        ex.set_num_threads(hardware_concurrency());
        ex.input("data", inputc._in);
        std::cerr << "input done" << std::endl;
        ex.extract("detection_out", inputc._out);
        std::cerr << "extract done" << std::endl;
        
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

        if (ad.has("nclasses"))
            _nclasses = ad.get("nclasses").get<int>();

        std::cerr << inputc._out.h << std::endl;
        for (int i = 0; i < inputc._out.h; i++) {
            const float* values = inputc._out.row(i);

            cats.push_back(this->_mlmodel.get_hcorresp(values[0]));
            probs.push_back(values[1]);

            APIData ad_bbox;
            std::cerr << "prob: " << values[1] << " / xmin: " << values[2] * cols << " / ymax: " << values[3] * rows << " / xmax: " << values[4] * cols << " / ymin: " << values[5] * rows << std::endl;
            ad_bbox.add("xmin",values[2] * cols);
            ad_bbox.add("ymax",values[3] * rows);
            ad_bbox.add("xmax",values[4] * cols);
            ad_bbox.add("ymin",values[5] * rows);
            bboxes.push_back(ad_bbox);
        }
        std::cerr << "bbox values loop finished" << std::endl;

        rad.add("uri", "1");
        rad.add("loss", 0.0);
        rad.add("probs", probs);
        rad.add("cats", cats);
        rad.add("bboxes", bboxes);

        std::cerr << "rad add done" << std::endl;

        vrad.push_back(rad);
        std::cerr << "vrad push back done" << std::endl;
        tout.add_results(vrad);
        std::cerr << "tout add results done" << std::endl;

        out.add("nclasses", _nclasses);
        out.add("bbox", true);
        out.add("roi", false);
        out.add("multibox_rois", false);

        std::cerr << "out add done" << std::endl;

        tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));

        std::cerr << "tout finalize done" << std::endl;

        out.add("status", 0);
        std::cerr << "predict finished" << std::endl;
    }

    template class NCNNLib<ImgNCNNInputFileConn,SupervisedOutput,NCNNModel>;
}
