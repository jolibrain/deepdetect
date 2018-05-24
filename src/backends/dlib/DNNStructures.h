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
