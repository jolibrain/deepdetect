/**
 * DeepDetect
 * copyright (c) 2018 Pixel Forensics, Inc.
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

#ifndef DLIBINPUTCONNS_H
#define DLIBINPUTCONNS_H

#include "imginputfileconn.h"
#include "opencv2/opencv.hpp"
#include "dlib/data_io.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/image_processing.h"
#include "dlib/opencv/to_open_cv.h"
#include "dlib/opencv/cv_image.h"


#include "inputconnectorstrategy.h"
#include "ext/base64/base64.h"

namespace dd {
    class DlibInputInterface {
    public:
        DlibInputInterface() {}

        DlibInputInterface(const DlibInputInterface &tii)
                : _dv(tii._dv) {}

        ~DlibInputInterface() {}

    public:
        // parameters common to all Dlib input connectors
        std::vector<dlib::matrix<dlib::rgb_pixel>> _dv;
        std::vector<dlib::matrix<dlib::rgb_pixel>> _dv_test;
        std::unordered_map<std::string,std::pair<int,int>> _imgs_size; /**< image sizes, used in detection. */
    };

    class ImgDlibInputFileConn : public ImgInputFileConn, public DlibInputInterface {
    public:
        ImgDlibInputFileConn()
                : ImgInputFileConn() {
            reset_dv();
        }

        ImgDlibInputFileConn(const ImgDlibInputFileConn &i)
                : ImgInputFileConn(i), DlibInputInterface(i){}

        ~ImgDlibInputFileConn() {}

        int channels() const {
            if (_bw) return 1;
            else return 3; // RGB
        }

        int height() const {
            return _height;
        }

        int width() const {
            return _width;
        }

        int batch_size() const {
            if (!_dv.empty())
                return _dv.size();
            else return ImgInputFileConn::batch_size();
        }

        int test_batch_size() const {
            if (!_dv_test.empty())
                return _dv_test.size();
            else return ImgInputFileConn::test_batch_size();
        }

        void init(const APIData &ad) {
            ImgInputFileConn::init(ad);
        }

        void transform(const APIData &ad) {
            try {
                ImgInputFileConn::transform(ad);
            }
            catch (InputConnectorBadParamException &e) {
                throw;
            }
	    // ids
	    bool set_ids = false;
	    if (this->_ids.empty())
	      set_ids = true;
            for (size_t i = 0; i < _images.size(); i++) {
                dlib::matrix<dlib::rgb_pixel> inputImg;
                if (_bw) {
                    dlib::assign_image(inputImg, dlib::cv_image<unsigned char>(_images.at(i)));
                } else {
                    dlib::assign_image(inputImg, dlib::cv_image<dlib::rgb_pixel>(_images.at(i)));
                }
                _dv.push_back(inputImg);
		if (set_ids)
		  this->_ids.push_back(_uris.at(i));
                _imgs_size.insert(std::pair<std::string,std::pair<int,int>>(this->_uris.at(i),this->_images_size.at(i)));
            }
            this->_images.clear();
            this->_images_size.clear();
        }

        std::vector<dlib::matrix<dlib::rgb_pixel>> get_dv(const int &num) {
            int i = 0;
            std::vector<dlib::matrix<dlib::rgb_pixel>> dv;
            while(_dt_vit!=_dv.end() && i < num) {
                dv.push_back((*_dt_vit));
                ++i;
                ++_dt_vit;
            }
            return dv;
        }

        void reset_dv()
        {
            _dt_vit = _dv.begin();
        }

    public:
        std::vector<dlib::matrix<dlib::rgb_pixel>>::const_iterator _dt_vit;

    };

}

#endif

