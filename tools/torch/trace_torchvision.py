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

import torch
import torchvision.models as M

import sys
import os
import argparse
import logging

parser = argparse.ArgumentParser(description="Trace image processing models from torchvision")
parser.add_argument('models', type=str, nargs='*', help="Models to trace.")
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

    def forward(self, x):
        """
        Input format: one tensor of dimensions [batch size, channel count, width, height]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        return self.model(l_x)


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
    "fasterrcnn_resnet50_fpn": M.detection.fasterrcnn_resnet50_fpn,
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

    kwargs = {}
    if args.num_classes:
        logging.info("Using num_classes = %d" % args.num_classes)
        kwargs["num_classes"] = args.num_classes

    logging.info("Exporting model %s %s", mname, "(pretrained)" if args.pretrained else "")
    model = model_classes[mname](pretrained=args.pretrained, progress=args.verbose, **kwargs)

    if args.to_dd_native:
        # Make model NativeModuleWrapper compliant
        model = Wrapper(model)

    model.eval()

    if mname in detection_model_classes:
        detect_model = DetectionModel(model)
        script_module = torch.jit.script(detect_model)
    else:
        # TODO try scripting instead of tracing
        example = torch.rand(1, 3, 224, 224)
        script_module = torch.jit.trace(model, example)

    filename = os.path.join(args.output_dir, mname + ("-pretrained" if args.pretrained else "") + ".pt")
    logging.info("Saving to %s", filename)
    script_module.save(filename)

logging.info("Done")
