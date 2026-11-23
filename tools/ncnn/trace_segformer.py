"""Trace the ADE20K SegFormer-B0 inference logits at 512x512 for PNNX."""

import argparse
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image).logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination TorchScript file")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512",
                        help="Hugging Face model ID or local checkpoint directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = Logits(SegformerForSemanticSegmentation.from_pretrained(args.model)).eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.zeros(1, 3, 512, 512))
        traced.save(str(args.output))


if __name__ == "__main__":
    main()
