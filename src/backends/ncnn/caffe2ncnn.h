#ifndef CAFFE2NCNN_H
#define CAFFE2NCNN_H

int convert_caffe_to_ncnn(bool ocr, const char *caffeproto,
                          const char *caffemodel, const char *ncnn_prototxt,
                          const char *ncnn_modelbin, int quantize_level,
                          const char *int8scale_table_path);

#endif
