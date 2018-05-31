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

#ifndef DLIBDNNSTRUCTURES_H
#define DLIBDNNSTRUCTURES_H

#include <dlib/dnn.h>


// --------      Common layers        ------- //
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<
dlib::affine<
        con5d<
                32, dlib::relu<
                    dlib::affine<
                    con5d<
                            32, dlib::relu<
                                dlib::affine<
                                con5d<16,SUBNET>>>>>>>>>;


// ------- Obj detect deep neural net ------- //

template <typename SUBNET> using rconObj5  = dlib::relu<dlib::affine<con5<55,SUBNET>>>;

using net_type_objDetector = dlib::loss_mmod<dlib::con<1,9,9,1,1,rconObj5<rconObj5<rconObj5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;


//// ------- Face detect deep neural net ------- //
template <typename SUBNET> using rconFace5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type_faceDetector = dlib::loss_mmod<dlib::con<1,9,9,1,1,rconFace5<rconFace5<rconFace5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

template<class T> using net_type = dlib::loss_mmod<T>;

#endif
