#!/usr/bin/env python

import sys
import numpy
import cPickle
import argparse
from caffe2.proto import caffe2_pb2
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
import detectron.core.test_engine as infer_engine

import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops()

# Hardcoded values
class Constants:

    ### Defined by Detectron

    # In Detectron/tools/convert_pkl_to_pb.py
    nms_outputs = ['score_nms', 'bbox_nms', 'class_nms'] # (see add_bbox_ops)
    im_info = 'im_info' # (see _prepare_blobs)
    # In Detectron/detectron/core/test.py # (see im_detect_mask)
    mask_rois = 'mask_rois'
    mask_pred = 'mask_fcn_probs'
    # In Detectron/detectron/modeling/FPN.py (see add_multilevel_roi_blobs)
    idx_restore_suffix = '_idx_restore_int32'
    @staticmethod
    def fpn_level_suffix(level): return '_fpn' + str(level)

    ### Defined by Deepdetect

    # In deepdetect/src/backends/caffe2/nettools/internal.h
    batch_splits_suffix = '_batch_splits'
    # In deepdetect/src/backends/caffe2/nettools/devices_and_operators.cc
    nms_outputs.append(
        nms_outputs[0] + batch_splits_suffix
    ) # (see ensure_box_nms_is_batchable)

    ### Can be set to anything

    main_output = 'masks'

def add_custom_op(net, name, inputs, outputs, **args):
    op = net.op.add()
    op.type = name
    op.input[:] = inputs
    op.output[:] = outputs
    for k,v in args.items():
        arg = op.arg.add()
        arg.name = k
        if isinstance(v, int):
            arg.i = v
        elif isinstance(v, float):
            arg.f = v
        elif type(v) in (list, tuple, numpy.ndarray):
            assert len(v)
            if type(v[0]) in (int, numpy.int32):
                arg.ints[:] = v
            elif type(v[0]) in (float, numpy.float32):
                arg.floats[:] = v
            else:
                raise RuntimeError('Unknown type : {}'.format(type(v[0])))
        else:
            raise RuntimeError('Unknown type : {}'.format(type(v)))

def main():

    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, dest='yml', help='cfg model file')
    parser.add_argument('--wts', required=True, dest='pkl', help='weights model file')
    parser.add_argument('--out_dir', required=True, dest='out', help='output directory')
    args = parser.parse_args()

    # Load
    merge_cfg_from_file(args.yml)
    mask_net = infer_engine.initialize_model_from_cfg(args.pkl).mask_net._net
    init_net = caffe2_pb2.NetDef()
    net = caffe2_pb2.NetDef()

    # Create the initialization net
    blobs = cPickle.load(open(args.pkl))['blobs']
    externals = set(c2_utils.UnscopeName(inp) for inp in mask_net.external_input)
    for name in set(blobs.keys()).intersection(externals):
        blob = blobs[name]
        add_custom_op(init_net, 'GivenTensorFill', [], [name],
                      values = blob.flatten(),
                      shape = blob.shape)

    # Pre-process the ROIs
    add_custom_op(net, 'BBoxToRoi',
                [Constants.nms_outputs[1], Constants.im_info, Constants.nms_outputs[3]],
                [Constants.mask_rois])
    # Group the ROIs based on their FPN level
    if cfg.FPN.MULTILEVEL_ROIS:
        outputs = [Constants.mask_rois + Constants.idx_restore_suffix]
        for level in range(cfg.FPN.ROI_MIN_LEVEL, cfg.FPN.ROI_MAX_LEVEL + 1):
            outputs.append(Constants.mask_rois + Constants.fpn_level_suffix(level))
        add_custom_op(net, 'MultiLevelRoi', [Constants.mask_rois], outputs,
                    min_level = cfg.FPN.ROI_MIN_LEVEL,
                    canon_scale = cfg.FPN.ROI_CANONICAL_SCALE,
                    canon_level = cfg.FPN.ROI_CANONICAL_LEVEL)

    # Generate the masks
    net.op.extend(mask_net.op)

    # Post-process the masks
    boxes = Constants.nms_outputs[:-1]
    infos = [Constants.im_info, Constants.nms_outputs[-1]]
    add_custom_op(net, 'SegmentMask',
                  boxes + [Constants.mask_pred] + infos,
                  boxes + [Constants.main_output] + infos,
                  thresh_bin = cfg.MRCNN.THRESH_BINARIZE)

    # Export
    open(args.out + '/predict_net.pb', 'wb').write(net.SerializeToString())
    open(args.out + '/init_net.pb', 'wb').write(init_net.SerializeToString())

    return 0

if __name__ == "__main__":
    exit(main())
