
#ifdef USE_CAFFE
#include "caffeinputconns.h"
#endif

#ifdef USE_TF
#include "backends/tf/tfinputconns.h"
#endif

#ifdef USE_DLIB
#include "backends/dlib/dlibinputconns.h"
#endif

#ifdef USE_NCNN
#include "backends/ncnn/ncnninputconns.h"
#endif

#ifdef USE_CAFFE2
#include "backends/caffe2/caffe2inputconns.h"
#endif

#ifdef USE_TENSORRT
#include "backends/tensorrt/tensorrtinputconns.h"
#endif

#ifdef USE_TORCH
#include "backends/torch/torchinputconns.h"
#endif

#ifdef USE_XGBOOST
#include "backends/xgb/xgbinputconns.h"
#endif

#ifdef USE_TSNE
#include "backends/tsne/tsneinputconns.h"
#endif
