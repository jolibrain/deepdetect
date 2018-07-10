/**
 * DeepDetect
 * Copyright (c) 2018 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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

#include <string>
#include "dliblib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

namespace dd {

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::DlibLib(const DlibModel &cmodel)
            :MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, DlibModel>(cmodel) {
        this->_libname = "dlib";
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::DlibLib(DlibLib &&cl) noexcept
            :MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, DlibModel>(std::move(cl)) {
        this->_libname = "dlib";
        _net_type = cl._net_type;
        this->_mltype = "detection/classification (" + _net_type + ")";
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~DlibLib() {
        std::lock_guard <std::mutex> lock(_net_mutex);
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &ad) {
        if (ad.has("model_type")) {
            _net_type = ad.get("model_type").get<std::string>();
        }

        if (_net_type.empty() || (_net_type != "obj_detector" && _net_type != "face_detector")) {
            throw MLLibBadParamException("Must specify model type (obj_detector or face_detector)");
        }

        this->_mlmodel.read_from_repository(this->_mlmodel._repo, this->_logger);
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) {
        // NOT IMPLEMENTED
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad,
                                                                                    APIData &out) {
        // NOT IMPLEMENTED
        return 0;
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::test(const APIData &ad,
                                                                                    APIData &out) {
        // NOT IMPLEMENTED
    }

    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int DlibLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::predict(const APIData &ad,
                                                                                      APIData &out) {
        std::lock_guard <std::mutex> lock(_net_mutex);

        APIData ad_output = ad.getobj("parameters").getobj("output");

        double confidence_threshold = 0.0;
        if (ad_output.has("confidence_threshold")) {
            confidence_threshold = ad_output.get("confidence_threshold").get<double>();
        }
        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy tout;
        APIData cad = ad;
        cad.add("model_repo", this->_mlmodel._repo);
        try {
            inputc.transform(cad);
        }
        catch (std::exception &e) {
            throw;
        }

        APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
        int batch_size = inputc.batch_size();
        if (ad_mllib.has("test_batch_size")) {
            batch_size = ad_mllib.get("test_batch_size").get<int>();
        }

        bool bbox = false;
        if (ad_output.has("bbox") && ad_output.get("bbox").get<bool>()) {
            bbox = true;
        }

        const std::string modelFile = this->_mlmodel._modelName;
        if (modelFile.empty()) {
            throw MLLibBadParamException("No pre-trained model found in model repository");
        }
        this->_logger->info("predict: using modelFile dir={}", modelFile);

        // Load the model into memory if not already
        if (!_modelLoaded) {
            this->_logger->info("predict: loading model into memory ({})", modelFile);
            if (_net_type.empty()) {
                throw MLLibBadParamException("Net type not specified");
            } else if (_net_type == "obj_detector") {
                dlib::deserialize(this->_mlmodel._modelName) >> _objDetector;
            } else if (_net_type == "face_detector") {
                dlib::deserialize(this->_mlmodel._modelName) >> _faceDetector;
            } else {
                throw MLLibBadParamException("Unrecognized net type: " + _net_type);
            }
            _modelLoaded = true;
        }

        // vector for storing  the outputAPI of the file
        std::vector <APIData> vrad;
        inputc.reset_dv();
        int idoffset = 0;
        while (true) {
            std::vector <dlib::matrix<dlib::rgb_pixel>> dv = inputc.get_dv(batch_size);
            if (dv.empty()) break;

            // running the loaded model and saving the generated output
            std::chrono::time_point <std::chrono::system_clock> tstart = std::chrono::system_clock::now();
            std::vector <std::vector<dlib::mmod_rect>> detections;
            if (_net_type == "obj_detector") {
                try {
                    detections = _objDetector(dv, batch_size);
                } catch (dlib::error &e) {
                    throw MLLibInternalException(e.what());
                }
            } else if (_net_type == "face_detector") {
                try {
                    detections = _faceDetector(dv, batch_size);
                } catch (dlib::error &e) {
                    throw MLLibInternalException(e.what());
                }
            } else {
                throw MLLibBadParamException("Unrecognized net type: " + _net_type);
            }
            std::chrono::time_point <std::chrono::system_clock> tstop = std::chrono::system_clock::now();
            this->_logger->info("predict: forward pass time={}",
                                std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count());
            APIData rad;

            for (size_t i = 0; i < dv.size(); i++) {
                size_t height = dv[i].nr(), width = dv[i].nc(); // nr() is number of rows, nc() is number of columns
                std::string uri = inputc._ids.at(idoffset + i);
                rad.add("uri", uri);
                auto foundImg = inputc._imgs_size.find(uri);
                int rows = 1;
                int cols = 1;
                if (foundImg != inputc._imgs_size.end()) {
                    // original image size
                    rows = foundImg->second.first;
                    cols = foundImg->second.second;
                } else {
                    this->_logger->error("couldn't find original image size for {}",uri);
                }
                std::vector<double> probs;
                std::vector <std::string> cats;
                std::vector <APIData> bboxes;
                this->_logger->info("[Input {}] Found {} objects", i, detections[i].size());
                for (auto &d : detections[i]) {
                    this->_logger->info("Found obj: {} - {} ({})", d.label, d.detection_confidence, d.rect);

                    if (d.detection_confidence < confidence_threshold) continue; // Skip if it doesn't pass the conf threshold

                    probs.push_back(d.detection_confidence);
                    cats.push_back(d.label);

                    if (bbox) {
                        // bbox can be formed with d.rect.left()/top()/right()/bottom()
                        APIData ad_bbox;
                        ad_bbox.add("xmin", std::round((static_cast<double>(d.rect.left()) / static_cast<double>(width)) * cols));
                        ad_bbox.add("ymax", std::round((static_cast<double>(height - d.rect.top()) / static_cast<double>(height)) * rows));
                        ad_bbox.add("xmax", std::round((static_cast<double>(d.rect.right()) / static_cast<double>(width)) * cols));
                        ad_bbox.add("ymin", std::round((static_cast<double>(height - d.rect.bottom()) / static_cast<double>(height)) * rows));
                        bboxes.push_back(ad_bbox);
                    }


                }
                rad.add("probs", probs);
                rad.add("cats", cats);
                rad.add("loss", 0.0);
                if (bbox) rad.add("bboxes", bboxes);
                vrad.push_back(rad);
            }
            idoffset += dv.size();
        } // end prediction loop over batches
        tout.add_results(vrad);
        out.add("bbox", bbox);
        tout.finalize(ad.getobj("parameters").getobj("output"), out, static_cast<MLModel *>(&this->_mlmodel));
        out.add("status", 0);
        return 0;
    }


    template
    class DlibLib<ImgDlibInputFileConn, SupervisedOutput, DlibModel>;
}
