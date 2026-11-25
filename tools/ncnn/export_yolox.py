"""Export the fixed 640x640 decoded YOLOX-S head for the C++ example."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch"))
from yolox_ncnn import yolox_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="destination ONNX file")
    parser.add_argument("--classes", type=int, default=80, help="foreground classes (COCO: 80)")
    args = parser.parse_args()
    if args.classes < 1:
        parser.error("--classes must be positive")

    torch.set_num_threads(1)
    # The DeepDetect provider counts background in nclasses.
    config = {"service": {"mllib": {"nclasses": args.classes + 1}}, "height": 640, "width": 640}
    model, inputs, names = yolox_s([args.checkpoint], config)
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(args.output), **names, opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()
