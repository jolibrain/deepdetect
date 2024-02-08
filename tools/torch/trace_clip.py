#!/usr/bin/env python3

import torch, transformers, torchvision
import numpy as np
from torch import nn
from transformers import CLIPVisionModelWithProjection, TensorType
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from typing import Tuple
import argparse, logging, os

parser = argparse.ArgumentParser(
    description="Trace CLIP-based vision models from pytorch-transformers (only tested on openai/clip-vit-large-patch14-336 and its derivatives)"
)
parser.add_argument(
    "--model",
    type=str,
    help="Model to trace",
    default="openai/clip-vit-large-patch14-336",
)
parser.add_argument(
    "--cache-dir",
    type=str,
    help="Cache dir for HuggingFace models. Leave unset to use default",
    default=None,
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Set logging level to INFO",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    help="Output directory for traced models",
    default=".",
)
parser.add_argument(
    "--script-wrapper",
    action="store_true",
    help="Script the entire wrapper instead of just the transforms",
)
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f"pytorch version {torch.__version__}, from {torch.__file__}")
logging.info(
    f"transformers version {transformers.__version__}, from {transformers.__file__}"
)
logging.info(
    f"torchvision version {torchvision.__version__}, from {torchvision.__file__}"
)


class CLIPWrapper(nn.Module):
    def __init__(self, visionmodel):
        super(CLIPWrapper, self).__init__()
        self.visionmodel = visionmodel
        self.visionmodel.eval()

        if args.script_wrapper:
            self.visionmodel = torch.jit.trace(
                self.visionmodel.forward,
                example_kwarg_inputs={"pixel_values": torch.rand([1, 3, 336, 336])},
            )
            self.transforms = torch.jit.script(
                nn.Sequential(
                    T.ConvertImageDtype(torch.float),
                    T.Resize(
                        size=[
                            336,
                        ],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),  # Resize to 336 on shortest edge
                    T.CenterCrop([336, 336]),  # Center crop a square of 336x336
                    T.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                )
            )
        else:
            self.transforms = torch.jit.script(
                nn.Sequential(
                    T.ConvertImageDtype(torch.float),
                    T.Resize(
                        size=[
                            336,
                        ],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),  # Resize to 336 on shortest edge
                    T.CenterCrop([336, 336]),  # Center crop a square of 336x336
                    T.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                )
            )

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.transforms(x)
            return self.visionmodel(pixel_values=x)[0]


# use cache dir if you want to specify a custom huggingface cache dir
visionmodel = CLIPVisionModelWithProjection.from_pretrained(
    args.model, torchscript=True, cache_dir=args.cache_dir
)

model = CLIPWrapper(visionmodel)
model.eval()

if args.script_wrapper:
    logging.info(
        f"Scripting wrapper (underlying visionmodel.forward method will still be traced)"
    )
    traced_model = torch.jit.script(model, torch.rand([1, 3, 1366, 1024]))
    outputfilename = (
        f"{os.path.join(args.output_dir, args.model.replace('/', '-'))}-scripted.pt"
    )
else:
    logging.info(f"Tracing wrapper (underlying transforms will still be scripted)")
    traced_model = torch.jit.trace(model, torch.rand([1, 3, 1366, 1024]))
    outputfilename = (
        f"{os.path.join(args.output_dir, args.model.replace('/', '-'))}-traced.pt"
    )

logging.info(f"Saving to {outputfilename}")
torch.jit.save(traced_model, outputfilename)
logging.info("Done")

# To use in DeepDetect, use something like the following to create the service:
# curl -X PUT "http://localhost:8080/services/myclipmodel" -d '{
#   "mllib":"torch",
#   "description":"myclipmodel",
#   "type":"supervised",
#   "parameters":{
#     "input":{
#       "connector":"image",
#       "height":336,
#       "width":336,
#       "rgb":true
#     },
#     "mllib":{
#       "concurrent_predict":false
#     }
#   },
#   "model":{
#     "repository":"/opt/models/myclipmodel/"
#   }
# }'

# and something like the following to get the image embedding output:
# curl -X POST "http://localhost:8080/predict" -d '{
#   "service":"myclipmodel",
#   "parameters":{
#     "mllib":{
#       "extract_layer":"last"
#     }
#   },
#   "data":["test.jpg"]
# }'
