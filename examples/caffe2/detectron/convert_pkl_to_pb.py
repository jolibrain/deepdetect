#!/usr/bin/env python

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

import os
import sys
import copy
import numpy
import cPickle
import argparse
from caffe2.python import core
from caffe2.proto import caffe2_pb2
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.config import assert_and_infer_cfg
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
import detectron.utils.model_convert_utils as mutils
import detectron.datasets.dummy_datasets as dummy_datasets
import tools.convert_pkl_to_pb as convert_tools

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wts', required=True, help='weights model file')
    parser.add_argument('--cfg', required=True, help='cfg model file')
    parser.add_argument('--out_dir', required=True, help='output directory')
    parser.add_argument('--mask_dir', type=str, help='mask extension directory')
    parser.add_argument('--coco', action='store_true',
                        help='generate a corresp.txt file containing the 81 coco classes')
    parser.add_argument('--net_name', default='detectron',
                        type=str, help='optional name for the net')
    parser.add_argument('--fuse_af', default=1, type=int, help='1 to fuse_af')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        type=str, help='Device to run the model on')
    parser.add_argument('--net_execution_type', choices=['simple', 'dag'], default='simple',
                        type=str, help='caffe2 net execution type')
    parser.add_argument('--use_nnpack', default=1, type=int, help='Use nnpack for conv')
    parser.add_argument('opts', nargs=argparse.REMAINDER,
                        help='See detectron/core/config.py for all options')
    args = parser.parse_args()
    assert not args.device == 'gpu' and args.use_nnpack
    for key in {'cfg', 'wts', 'out_dir'}:
        setattr(args, key, os.path.abspath(getattr(args, key)))
    args.opts.extend(['TRAIN.WEIGHTS', args.wts, 'TEST.WEIGHTS', args.wts, 'NUM_GPUS', 1])
    return args

def save_model(net, init_net, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'predict_net.pb'), 'wb') as f:
        f.write(net.SerializeToString())
    with open(os.path.join(path, 'init_net.pb'), 'wb') as f:
        f.write(init_net.SerializeToString())

def convert_main_net(args, main_net, blobs):
    net = core.Net('')
    net.Proto().op.extend(copy.deepcopy(main_net.Proto().op))
    net.Proto().external_input.extend(copy.deepcopy(main_net.Proto().external_input))
    net.Proto().external_output.extend(copy.deepcopy(main_net.Proto().external_output))
    net.Proto().type = args.net_execution_type
    net.Proto().num_workers = 1 if args.net_execution_type == 'simple' else 4
    convert_tools.convert_net(args, net.Proto(), blobs)
    convert_tools.add_bbox_ops(args, net, blobs)
    if args.fuse_af:
        print ('Fusing affine channel...')
        net, blobs = mutils.fuse_net_affine(net, blobs)
    if args.use_nnpack:
        mutils.update_mobile_engines(net.Proto())
    empty_blobs = ['data', 'im_info']
    init_net = convert_tools.gen_init_net(net, blobs, empty_blobs)
    if args.device == 'gpu':
        [net, init_net] = convert_tools.convert_model_gpu(args, net, init_net)
    net.Proto().name = args.net_name
    init_net.Proto().name = args.net_name + '_init'
    save_model(net.Proto(), init_net.Proto(), args.out_dir)

def convert_mask_net(args, mask_net):

    # Initialization net
    init_net = caffe2_pb2.NetDef()
    net = caffe2_pb2.NetDef()
    blobs = cPickle.load(open(args.wts))['blobs']
    externals = set(c2_utils.UnscopeName(inp) for inp in mask_net.Proto().external_input)
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
    net.op.extend(mask_net.Proto().op)

    # Post-process the masks
    add_custom_op(net, 'SegmentMask',
                  Constants.nms_outputs[1:-1] + [Constants.mask_pred, Constants.im_info],
                  [Constants.main_output, Constants.im_info],
                  thresh_bin = cfg.MRCNN.THRESH_BINARIZE)

    net.name = args.net_name + '_mask'
    init_net.name = args.net_name + '_mask_init'
    save_model(net, init_net, args.mask_dir)

def main():
    args = parse_args()
    merge_cfg_from_file(args.cfg)
    merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    model, blobs = convert_tools.load_model(args)
    convert_main_net(args, model.net, blobs)
    if args.mask_dir:
        convert_mask_net(args, model.mask_net)
    if args.coco:
        classes = dummy_datasets.get_coco_dataset().classes
        corresp = '\n'.join('{} {}'.format(i, classes[i]) for i, _ in enumerate(classes))
        with open(args.out_dir + '/corresp.txt', 'w') as f:
            f.write(corresp)
    return 0

if __name__ == '__main__':
    exit(main())
