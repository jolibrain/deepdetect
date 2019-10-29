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
        this->_mltype = "detection";
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

        if (_net_type.empty() ||
            (_net_type != "obj_detector" && _net_type != "face_detector" && _net_type != "face_feature_extractor")) {
            throw MLLibBadParamException("Must specify model type (obj_detector, face_detector, or face_feature_extractor)");
        }

        if (ad.has("shape_predictor")) {
            this->_mlmodel._hasShapePredictor = true;
        }

        this->_mlmodel.read_from_repository(this->_mlmodel._repo, this->_logger);
        // If provided, use this as the chip_size for the shape predictor. Default: 150
        if (ad.has("chip_size")) {
            _chip_size = ad.get("chip_size").get<int>();
        }
        // If provided, use this as the padding ratio for the shape predictor. Default: 0.25
        if (ad.has("padding")) {
            _padding = ad.get("padding").get<double>();
        }
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

        // If provided, use this as the chip_size for the shape predictor. Default: 150
        int chip_size = _chip_size;
        if (ad_mllib.has("chip_size")) {
            chip_size = ad_mllib.get("chip_size").get<int>();
        }
        double padding = _padding;
        // If provided, use this as the padding ratio for the shape predictor. Default: 0.25
        if (ad_mllib.has("padding")) {
            padding = ad_mllib.get("padding").get<double>();
        }

        const std::string modelFile = this->_mlmodel._modelName;
        if (modelFile.empty()) {
            throw MLLibBadParamException("No pre-trained model found in model repository");
        }
        const std::string shapePredictorFile = this->_mlmodel._shapePredictorName;

        this->_logger->info("predict: using modelFile dir={}", modelFile);

        // Load the model into memory if not already
        if (!_modelLoaded) {
            this->_logger->info("predict: loading model into memory ({})", modelFile);
            if (_net_type.empty()) {
                throw MLLibBadParamException("Net type not specified");
            } else if (_net_type == "obj_detector") {
                dlib::deserialize(modelFile) >> _objDetector;
            } else if (_net_type == "face_detector") {
                dlib::deserialize(modelFile) >> _faceDetector;
            } else if (_net_type == "face_feature_extractor") {
                dlib::deserialize(modelFile) >> _faceFeatureExtractor;
            } else {
                throw MLLibBadParamException("Unrecognized net type: " + _net_type);
            }
            if (!shapePredictorFile.empty()) {
                dlib::deserialize(shapePredictorFile) >> _shapePredictor;
                this->_logger->info("predict: loaded shape predictor into memory ({})", shapePredictorFile);
            }
            _modelLoaded = true;
        }

        // vector for storing  the outputAPI of the file
        std::vector <APIData> vrad;
        std::vector<cv::Mat> cropped_imgs;
        std::vector<std::string> cids;
        inputc.reset_dv();
        int idoffset = 0;
        while (true) {
            std::vector <dlib::matrix<dlib::rgb_pixel>> dv = inputc.get_dv(batch_size);
            if (dv.empty()) break;

            // running the loaded model and saving the generated output
            std::chrono::time_point <std::chrono::system_clock> tstart = std::chrono::system_clock::now();
            std::vector <std::vector<dlib::mmod_rect>> detections;
            std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
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
            } else if (_net_type == "face_feature_extractor"){
                try {
                    face_descriptors = _faceFeatureExtractor(dv, batch_size);
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
                if (_net_type == "face_feature_extractor") {
                    // Only for feature extractor models
                    this->_logger->info("[Input {}] Extracted feature representation of size {}", i, face_descriptors[i].size());
                    std::vector<double> vals(face_descriptors[i].begin(), face_descriptors[i].end());
                    rad.add("vals",vals);
                } else {
                    // Only for detector-type models
                    this->_logger->info("[Input {}] Found {} objects", i, detections[i].size());
                    for (size_t j=0; j < detections[i].size(); j++) {
                        auto d = detections[i][j];
                        this->_logger->info("Found obj: {} - {} ({})", d.label, d.detection_confidence, d.rect);

                        if (d.detection_confidence < confidence_threshold)
                            continue; // Skip if it doesn't pass the conf threshold

                        probs.push_back(d.detection_confidence);
                        cats.push_back(d.label);

                        if (bbox) {
                            // bbox can be formed with d.rect.left()/top()/right()/bottom()
                            APIData ad_bbox;
                            ad_bbox.add("xmin", std::round(
                                    (static_cast<double>(d.rect.left()) / static_cast<double>(width)) * cols));
                            ad_bbox.add("ymax", std::round(
                                    (static_cast<double>(height - d.rect.top()) / static_cast<double>(height)) * rows));
                            ad_bbox.add("xmax", std::round(
                                    (static_cast<double>(d.rect.right()) / static_cast<double>(width)) * cols));
                            ad_bbox.add("ymin", std::round(
                                    (static_cast<double>(height - d.rect.bottom()) / static_cast<double>(height)) *
                                    rows));
                            bboxes.push_back(ad_bbox);

                            if (ad.has("chain") && ad.get("chain").get<bool>() && !shapePredictorFile.empty()) {
                                auto shape = _shapePredictor(dv[i], d.rect);
                                dlib::matrix<dlib::rgb_pixel> r;
                                dlib::extract_image_chip(dv[i], dlib::get_face_chip_details(shape,chip_size,padding), r);
                                cv::Mat cropped_img = dlib::toMat(r);
                                cropped_imgs.push_back(cropped_img);
                                std::string bboxstr = uri+"bbox"+std::to_string(j);
                                std::string bbox_id = std::to_string(std::hash<std::string>{}(bboxstr));
                                cids.push_back(bbox_id);
                            }
                        }
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
        if (ad.has("chain") && ad.get("chain").get<bool>()) {
            if (!shapePredictorFile.empty()) {
                APIData chain_input;
                chain_input.add("imgs", cropped_imgs);
                chain_input.add("imgs_size",reinterpret_cast<ImgDlibInputFileConn*>(&inputc)->_images_size);
                chain_input.add("cids", cids);
                out.add("input", chain_input);
            } else if (typeid(inputc) == typeid(ImgDlibInputFileConn)) {
                APIData chain_input;
                if (!reinterpret_cast<ImgDlibInputFileConn*>(&inputc)->_orig_images.empty())
                    chain_input.add("imgs",reinterpret_cast<ImgDlibInputFileConn*>(&inputc)->_orig_images);
                else
                    chain_input.add("imgs",reinterpret_cast<ImgDlibInputFileConn*>(&inputc)->_images);
                chain_input.add("imgs_size",reinterpret_cast<ImgDlibInputFileConn*>(&inputc)->_images_size);
                out.add("input",chain_input);
            }
        }
        out.add("status", 0);
        return 0;
    }

    template class DlibLib<ImgDlibInputFileConn, SupervisedOutput, DlibModel>;
    template class DlibLib<ImgDlibInputFileConn, UnsupervisedOutput, DlibModel>;
}
