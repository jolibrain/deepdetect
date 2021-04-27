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

import torch

# Use dede torchvision until randperm bug is solved
# see https://github.com/pytorch/vision/issues/3469
import importlib
spec = importlib.util.find_spec("torchvision")
torch.ops.load_library(os.path.join(os.path.dirname(spec.origin), "_C.so"))
sys.path = [os.path.join(os.path.dirname(__file__), "../../build/pytorch_vision/src/pytorch_vision/")] + sys.path

import torchvision.models as M

parser = argparse.ArgumentParser(description="Trace image processing models from torchvision")
parser.add_argument('models', type=str, nargs='*', help="Models to trace.")
parser.add_argument('--backbone', type=str, help="Backbone for detection models")
parser.add_argument('--print-models', action='store_true', help="Print all the available models names and exit")
parser.add_argument('--to-dd-native', action='store_true', help="Prepare the model so that the weights can be loaded on native model with dede")
parser.add_argument('-a', "--all", action='store_true', help="Export all available models")
parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
parser.add_argument('-o', "--output-dir", default=".", type=str, help="Output directory for traced models")
parser.add_argument('-p', "--not-pretrained", dest="pretrained", action='store_false',
                    help="Whether the exported models should not be pretrained")
parser.add_argument('--cpu', action='store_true', help="Force models to be exported for CPU device")
parser.add_argument('--num_classes', type=int, help="Number of classes")

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

    def forward(self, x, bboxes = None, labels = None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, List[Dict[str, Tensor]]]
        """
        Input format: one tensor of dimensions [batch size, channel count, width, height]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        if self.training:
            assert bboxes is not None
            assert labels is not None
            l_targs = [{"boxes": bboxes[i], "labels": labels[i]} for i in range(x.shape[0])]
            losses, predictions = self.model(l_x, l_targs)

            # Sum of all losses for finetuning (as done in vision/references/detection/engine.py)
            losses = [l for l in losses.values()]
            loss = torch.zeros((1,), device=x.device, dtype=x.dtype)
            for i in range(len(losses)):
                loss += losses[i]
        else:
            losses, predictions = self.model(l_x)
            loss = torch.zeros((1,), device=x.device, dtype=x.dtype)

        return loss, predictions

def get_detection_input():
    """
    Sample input for detection models, usable for tracing or testing
    """
    return (
            torch.rand(1, 3, 224, 224),
            torch.Tensor([1, 1, 200, 200]).unsqueeze(0).unsqueeze(0),
            torch.full((1,1), 1).long()
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
    "resnext50_32x4d": M.resnext50_32x4d,
    "resnext101_32x8d": M.resnext101_32x8d,
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

    if detection:
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

            model = model_classes[mname](backbone, args.num_classes)
        else:
            if args.backbone:
                raise RuntimeError("--backbone is only supported with models \"fasterrcnn\" or \"retinanet\".")
            model = model_classes[mname](pretrained=args.pretrained, progress=args.verbose)

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

        detect_model = DetectionModel(model)
        detect_model.train()
        script_module = torch.jit.script(detect_model)

    else:
        kwargs = {}
        if args.num_classes:
            logging.info("Using num_classes = %d" % args.num_classes)
            kwargs["num_classes"] = args.num_classes

        model = model_classes[mname](pretrained=args.pretrained, progress=args.verbose, **kwargs)

        if args.to_dd_native:
            # Make model NativeModuleWrapper compliant
            model = Wrapper(model)

        model.eval()

        # TODO try scripting instead of tracing
        example = torch.rand(1, 3, 224, 224)
        script_module = torch.jit.trace(model, example)

    filename = os.path.join(args.output_dir, mname + ("-pretrained" if args.pretrained else "") + ".pt")
    logging.info("Saving to %s", filename)
    script_module.save(filename)

logging.info("Done")
