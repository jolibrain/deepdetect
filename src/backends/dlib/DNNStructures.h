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
// Download and uncompress pretrained model from: http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2
// Check https://github.com/davisking/dlib-models for more models

template <typename SUBNET> using rconObj5  = dlib::relu<dlib::affine<con5<55,SUBNET>>>;

using net_type_objDetector = dlib::loss_mmod<dlib::con<1,9,9,1,1,rconObj5<rconObj5<rconObj5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;


//// ------- Face detect deep neural net ------- //
// Download and uncompress pretrained model from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
template <typename SUBNET> using rconFace5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type_faceDetector = dlib::loss_mmod<dlib::con<1,9,9,1,1,rconFace5<rconFace5<rconFace5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;


// ------- Face recognition feature extraction deep neural net (ResNet) -------- //
// Download and uncompress pretrained model from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
template <template <int,template<typename>class,int,typename> class dlibblock, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<dlibblock<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class dlibblock, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<dlibblock<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using dlibblock  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<dlibblock,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<dlibblock,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using net_type_faceFeatureExtractor = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                                                         alevel0<
                                                                 alevel1<
                                                                         alevel2<
                                                                                 alevel3<
                                                                                         alevel4<
                                                                                                 dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                                                                                                 dlib::input_rgb_image_sized<150>
                                                                                 >>>>>>>>>>>>;

#endif
