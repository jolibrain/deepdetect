import torch
import timm
from timm.models import create_model, list_models
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", help="model name from timm")
parser.add_argument("--in-chans", help="number of input channels", type=int, default=3)
parser.add_argument(
    "--nclasses", help="number of output classes for the model", type=int, default=2
)
parser.add_argument(
    "--pretrained",
    help="whether to download the model weights if they exist",
    action="store_true",
)
parser.add_argument(
    "--no-head",
    help="whether to not export the model classification head",
    action="store_true",
)
parser.add_argument(
    "-a", "--all", help="whether to export all models", action="store_true"
)
parser.add_argument(
    "-o",
    "--output-dir",
    default=".",
    type=str,
    help="Output directory for traced models",
)
args = parser.parse_args()

list_timm_models = list_models()
if not args.all and not args.model_name in list_timm_models:
    print(list_timm_models)
    print("unknown model ", args.model_name)
    sys.exit()

if args.all:
    models = list_timm_models
else:
    models = [args.model_name]

nclasses = args.nclasses
if args.no_head:
    nclasses = 0

for m in models:
    try:
        model = create_model(
            m,
            num_classes=nclasses,
            in_chans=args.in_chans,
            pretrained=args.pretrained,
            scriptable=True,
            exportable=True,
        )
    except:
        print("failed creating model " + m + " with requested options")
        continue
    jit_model = None
    try:
        jit_model = torch.jit.script(model)
    except:
        print("failed JIT export of model " + m)
        continue

    out_model_name = m
    out_model_name += "_inchans" + str(args.in_chans)
    if args.no_head:
        out_model_name += "_nohead"
    if args.pretrained:
        out_model_name += "_pretrained"
    if nclasses > 0:
        out_model_name += "_cls" + str(nclasses)
    jit_model.save(args.output_dir + "/" + out_model_name + ".pt")
    print("saved model ", args.output_dir + "/" + out_model_name + ".pt")
