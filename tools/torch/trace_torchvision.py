#!/usr/bin/python3
"""
DeepDetect
Copyright (c) 2019 Jolibrain
Author: Louis Jean <ljean@etud.insa-toulouse.fr>

This file is part of deepdetect.

deepdetect is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

deepdetect is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
import argparse
import logging
from typing import Dict, List
from packaging import version

import torch
import torchvision
import torchvision.models as M
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

parser = argparse.ArgumentParser(description="Trace image processing models from torchvision")
parser.add_argument('models', type=str, nargs='*', help="Models to trace.")
parser.add_argument('--backbone', type=str, help="Backbone for detection models")
parser.add_argument('--print-models', action='store_true', help="Print all the available models names and exit")
parser.add_argument('--to-dd-native', action='store_true', help="Prepare the model so that the weights can be loaded on native model with dede")
parser.add_argument('--to-onnx', action="store_true", help="If specified, export to onnx instead of jit.")
parser.add_argument('--onnx_out', type=str, default="prob", help="Name of onnx output")
parser.add_argument('--weights', type=str, help="If not None, these weights will be embedded in the model before exporting")
parser.add_argument('-a', "--all", action='store_true', help="Export all available models")
parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
parser.add_argument('-o', "--output-dir", default=".", type=str, help="Output directory for traced models")
parser.add_argument('-p', "--not-pretrained", dest="pretrained", action='store_false',
                    help="Whether the exported models should not be pretrained")
parser.add_argument('--cpu', action='store_true', help="Force models to be exported for CPU device")
parser.add_argument('--num_classes', type=int, help="Number of classes")
parser.add_argument('--trace', action='store_true', help="Whether to trace model instead of scripting")
parser.add_argument('--batch_size', type=int, default=1, help="When exporting with fixed batch size, this will be the batch size of the model")
parser.add_argument('--img_width', type=int, default=224, help="Width of the image when exporting with fixed image size")
parser.add_argument('--img_height', type=int, default=224, help="Height of the image when exporting with fixed image size")

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)


class Wrapper(torch.nn.Module):
    """
    Dede native models are wrapped into a NativeModuleWrapper,
    so we mimic the structure here.
    """
    def __init__(self, wrapped):
        super(Wrapper, self).__init__()
        self.wrapped = wrapped

    def forward(self, x):
        return self.wrapped(x)


class DetectionModel(torch.nn.Module):
    """
    Adapt input and output of detection model to make it usable by dede.
    """
    def __init__(self, model):
        super(DetectionModel, self).__init__()
        self.model = model

    def forward(self, x, ids = None, bboxes = None, labels = None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Dict[str,Tensor], List[Dict[str, Tensor]]]
        """
        x: one image of dimensions [batch size, channel count, width, height]
        ids: one tensor of dimension [sum(n_bbox_i)] containing id of batch for
        each bbox.
        bbox: one tensor of dimension [sum(n_bbox_i), 4]
        labels: one tensor of dimension [sum(n_bbox_i)]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        if self.training:
            assert ids is not None
            assert bboxes is not None
            assert labels is not None

            l_targs : List[Dict[str, Tensor]] = []
            stop = 0

            for i in range(x.shape[0]):
                start = stop

                while stop < ids.shape[0] and ids[stop] == i:
                    stop += 1

                targ = {"boxes": bboxes[start:stop], "labels": labels[start:stop]}
                l_targs.append(targ)

            losses, predictions = self.model(l_x, l_targs)

            # Sum of all losses for finetuning (as done in vision/references/detection/engine.py)
            losses["total_loss"] = torch.sum(torch.stack(losses.values()))
        else:
            losses, predictions = self.model(l_x)

        return losses, predictions


class DetectionModel_PredictOnly(torch.nn.Module):
    """
    Adapt input and output of the model to make it exportable to
    ONNX
    """
    def __init__(self, model):
        super(DetectionModel_PredictOnly, self).__init__()
        self.model = model

    def forward(self, x):
        l_x = [x[i] for i in range(x.shape[0])]
        predictions = self.model(l_x)
        # To dede format
        pred_list = list()
        for i in range(x.shape[0]):
            pred_list.append(
                    torch.cat((
                        torch.full(predictions[i]["labels"].shape, i, dtype=float).unsqueeze(1),
                        predictions[i]["labels"].unsqueeze(1).float(),
                        predictions[i]["scores"].unsqueeze(1),
                        predictions[i]["boxes"]), dim=1))

        return torch.cat(pred_list)

def get_image_input(batch_size=1, img_width=224, img_height=224):
    return torch.rand(batch_size, 3, img_width, img_height)

def get_detection_input(batch_size=1, img_width=224, img_height=224):
    """
    Sample input for detection models, usable for tracing or testing
    """
    return (
            torch.rand(batch_size, 3, img_width, img_height),
            torch.arange(0, batch_size).long(),
            torch.Tensor([1, 1, 200, 200]).repeat((batch_size, 1)),
            torch.full((batch_size,), 1).long(),
    )

model_classes = {
    "alexnet": M.alexnet,
    "vgg11": M.vgg11,
    "vgg11_bn": M.vgg11_bn,
    "vgg13": M.vgg13,
    "vgg13_bn": M.vgg13_bn,
    "vgg16": M.vgg16,
    "vgg16_bn": M.vgg16_bn,
    "vgg19": M.vgg19,
    "vgg19_bn": M.vgg19_bn,
    "resnet18": M.resnet18,
    "resnet34": M.resnet34,
    "resnet50": M.resnet50,
    "resnet101": M.resnet101,
    "resnet152": M.resnet152,
    "squeezenet1_0": M.squeezenet1_0,
    "squeezenet1_1": M.squeezenet1_1,
    "densenet121": M.densenet121,
    "densenet169": M.densenet169,
    "densenet161": M.densenet161,
    "densenet201": M.densenet201,
    "googlenet": M.googlenet,
    "shufflenet_v2_x0_5": M.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": M.shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": M.shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": M.shufflenet_v2_x1_0,
    "mobilenet_v2": M.mobilenet_v2,
    "mobilenet_v3_small": M.mobilenet_v3_small,
    "mobilenet_v3_large": M.mobilenet_v3_large,
    "resnext50_32x4d": M.resnext50_32x4d,
    "resnext101_32x8d": M.resnext101_32x8d,
    "wide_resnet50_2": M.wide_resnet50_2,
    "mnasnet0_5": M.mnasnet0_5,
    "mnasnet0_75": M.mnasnet0_75,
    "mnasnet1_0": M.mnasnet1_0,
    "mnasnet1_3": M.mnasnet1_3
}
detection_model_classes = {
    "fasterrcnn": M.detection.FasterRCNN,
    "fasterrcnn_resnet50_fpn": M.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": M.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_mobilenet_v3_large_320_fpn": M.detection.fasterrcnn_mobilenet_v3_large_320_fpn,

    "retinanet": M.detection.RetinaNet,
    "retinanet_resnet50_fpn": M.detection.retinanet_resnet50_fpn,
}
model_classes.update(detection_model_classes)
segmentation_model_classes = {
    "fcn_resnet50": M.segmentation.fcn_resnet50,
    "fcn_resnet101": M.segmentation.fcn_resnet101,
    "deeplabv3_resnet50": M.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": M.segmentation.deeplabv3_resnet101,
    "deeplabv3_mobilenet_v3_large": M.segmentation.deeplabv3_mobilenet_v3_large,
    "lraspp_mobilenet_v3_large": M.segmentation.lraspp_mobilenet_v3_large
}
model_classes.update(segmentation_model_classes)


if args.all:
    args.models = model_classes.keys()

if args.print_models:
    print("*** Available models ***")
    for key in model_classes:
        print(key)
    sys.exit(0)
elif not args.models:
    sys.stderr.write("Please specify at least one model to be exported\n")
    sys.exit(-1)

device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
logging.info("Device: %s", device)

# An instance of your model.
for mname in args.models:
    if mname not in model_classes:
        logging.warn("model %s is unknown and will not be exported", mname)
        continue

    logging.info("Exporting model %s %s", mname, "(pretrained)" if args.pretrained else "")
    detection = mname in detection_model_classes
    segmentation = mname in segmentation_model_classes
    
    if detection:
        if "fasterrcnn" in mname and version.parse(torchvision.__version__) < version.parse("0.10.0"):
            raise RuntimeError("Fasterrcnn needs torchvision >= 0.10.0 (current = %s)" % torchvision.__version__)

        if mname in ["fasterrcnn", "retinanet"]:
            if args.backbone and args.backbone in model_classes:
                if "resnet" in args.backbone or "resnext" in args.backbone:
                    backbone = M.detection.backbone_utils.resnet_fpn_backbone(args.backbone, pretrained = args.pretrained)
                elif "mobilenet" in args.backbone:
                    backbone = M.detection.backbone_utils.mobilenet_backbone(args.backbone, pretrained = args.pretrained, fpn = True)
                else:
                    raise RuntimeError("Backbone not supported: %s. Supported backbones are resnet, resnext or mobilenet." % args.backbone)
            else:
                raise RuntimeError("Please specify a backbone for model %s" % mname)

            if args.pretrained:
                logging.warn("Pretrained models are not available for custom backbones. " +
                            "Output model (except the backbone) will be untrained.")
                args.pretrained = False

            model = model_classes[mname](backbone, args.num_classes)
        else:
            if args.backbone:
                raise RuntimeError("--backbone is only supported with models \"fasterrcnn\" or \"retinanet\".")
            model = model_classes[mname](pretrained=args.pretrained, pretrained_backbone=args.pretrained, progress=args.verbose)

            if args.num_classes:
                logging.info("Using num_classes = %d" % args.num_classes)

                if "fasterrcnn" in mname:
                    # get number of input features for the classifier
                    in_features = model.roi_heads.box_predictor.cls_score.in_features
                    # replace the pre-trained head with a new one
                    model.roi_heads.box_predictor = M.detection.faster_rcnn.FastRCNNPredictor(in_features, args.num_classes)
                elif "retinanet" in mname:
                    in_channels = model.backbone.out_channels
                    num_anchors = model.head.classification_head.num_anchors
                    # replace pretrained head
                    model.head = M.detection.retinanet.RetinaNetHead(in_channels, num_anchors, args.num_classes)

        if args.to_onnx:
            model = DetectionModel_PredictOnly(model)
            model.eval()
        else:
            model = DetectionModel(model)
            model.train()
            script_module = torch.jit.script(model)

        if args.num_classes is None:
            # TODO dont hard code this
            args.num_classes = 91

    else:
        kwargs = {}
        if args.num_classes and not segmentation:
            logging.info("Using num_classes = %d" % args.num_classes)
            kwargs["num_classes"] = args.num_classes

        model = model_classes[mname](pretrained=args.pretrained, progress=args.verbose, **kwargs)

        if segmentation and 'deeplabv3' in mname:
            model.classifier = DeepLabHead(2048, args.num_classes)
        
        if args.to_dd_native:
            # Make model NativeModuleWrapper compliant
            model = Wrapper(model)

        model.eval()

        # tracing or scripting model (default)
        if args.trace:
            example = get_image_input(args.batch_size, args.img_width, args.img_height) 
            script_module = torch.jit.trace(model, example)
        else:
            script_module = torch.jit.script(model)

    filename = os.path.join(
            args.output_dir,
            mname
            + ("-pretrained" if args.pretrained else "")
            + ("-" + args.backbone if args.backbone else "")
            + ("-cls" + str(args.num_classes) if args.num_classes else "")
            + ".pt")
    
    if args.weights:
        # load weights
        weights = torch.jit.load(args.weights).state_dict()
        
        if args.to_onnx:
            logging.info("Apply weights from %s to the onnx model" % args.weights)
            model.load_state_dict(weights, strict=True)
        else:
            logging.info("Apply weights from %s to the jit model" % args.weights)
            script_module.load_state_dict(weights, strict=True)

    if args.to_onnx:
        logging.info("Export model to onnx (%s)" % filename)
        # remove extension
        filename = filename[:-3] + ".onnx"
        example = get_image_input(args.batch_size, args.img_width, args.img_height) 

        # change for detection
        torch.onnx.export(
                model, example, filename,
                export_params=True, verbose=args.verbose,
                opset_version=11, do_constant_folding=True,
                input_names=["input"], output_names=[args.onnx_out])
        # dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}}
    else:
        logging.info("Saving to %s", filename)
        script_module.save(filename)

logging.info("Done")
