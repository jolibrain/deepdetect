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
parser.add_argument('-a', "--all", action='store_true', help="Export all available models")
parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
parser.add_argument('-o', "--output-dir", default=".", type=str, help="Output directory for traced models")
parser.add_argument('-p', "--not-pretrained", dest="pretrained", action='store_false', 
                    help="Whether the exported models should not be pretrained")
parser.add_argument('--cpu', action='store_true', help="Force models to be exported for CPU device")

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

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
    model = model_classes[mname](pretrained=args.pretrained, progress=args.verbose)
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)

    filename = os.path.join(args.output_dir, mname + ("-pretrained" if args.pretrained else "") + ".pt")
    logging.info("Saving to %s", filename)
    traced_script_module.save(filename)

logging.info("Done")