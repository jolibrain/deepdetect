"""
TODO:
    - investigate influence of num_queries / num_top_queries / num_denoising
    - test "discrete sampling" models if needed: https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch#models-of-discrete-sampling

    - why change eval_idx here: https://github.com/lyuwenyu/RT-DETR/blob/0b6972de10bc968045aba776ec1a60efea476165/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml#L17
      it means during training we do all layers, but not during eval: https://github.com/lyuwenyu/RT-DETR/blob/0b6972de10bc968045aba776ec1a60efea476165/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L272
      why not use less layers in this case?

    - the default config wants to use focal_loss: https://github.com/lyuwenyu/RT-DETR/blob/0b6972de10bc968045aba776ec1a60efea476165/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml#L8
      which is used here: https://github.com/lyuwenyu/RT-DETR/blob/0b6972de10bc968045aba776ec1a60efea476165/rtdetrv2_pytorch/src/zoo/rtdetr/matcher.py#L88
      but criterion always uses vfl: https://github.com/lyuwenyu/RT-DETR/blob/0b6972de10bc968045aba776ec1a60efea476165/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py#L69

CREATE SERVICE:
    {
        "parameters": {
            "input": {
                "connector": "image",
                "scale": 0.003921569,
                "rgb": true,
                ...
            },
            "mllib": {
                "template": "detr",
                "nclasses": 2,
                ...
            },
            ...
        },
        ...
    }


TRAIN SERVICE:
    {
        "parameters": {
            "mllib": {
                "solver": {
                    "base_lr": 1e-05,
                    "clip": true,
                    "clip_norm": 0.1,
                    ...
                },
                ...
            },
            ...
        },
        ...
    }
"""

import os
import sys
import argparse
import torch
import torchvision
import math


class WrappedRTDETR(torch.nn.Module):
    def __init__(self, cfg, nms_threshold):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.criterion = cfg.criterion
        self.nms_threshold = nms_threshold

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def dd_targets_to_detr_targets(self, ids, bboxes, labels, target_sizes):
        # type: (Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor)
        assert ids is not None
        assert bboxes is not None
        assert labels is not None

        # convert DD bboxes to DETR bboxes instead, because of the L1 loss here:
        # https://github.com/facebookresearch/detr/blob/main/models/detr.py#L153

        # assume all images in the batch are the same size
        img_w, img_h = target_sizes[0].unbind(0)

        # convert to [xc, yc, w, h] format
        bboxes = self.box_xyxy_to_cxcywh(bboxes)

        # and to relative [0, 1] coordinates
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0).to(bboxes.device)
        bboxes = bboxes / scale_fct

        # remove no bbox
        valids = labels > 0
        ids = ids[valids]
        bboxes = bboxes[valids]
        labels = labels[valids]

        # convert ids, bboxes, labels to DETR targets
        # DD uses ids, DETR expects lists of boxes, labels
        detr_targets: List[Dict[str, torch.Tensor]] = []
        # DETR classes start at 0
        labels -= 1
        batch_size = target_sizes.shape[0]
        count = torch.bincount(ids, minlength=batch_size)
        start = 0
        for i in range(batch_size):
            stop = start + count[i]
            target = {
                "labels": labels[start:stop],
                "boxes": bboxes[start:stop],
            }
            detr_targets.append(target)
            start = stop

        return detr_targets

    def detr_outputs_to_dd_outputs(self, outputs, target_sizes):
        # type: (Dict[str, Tensor], Tensor)
        labels, boxes, scores = self.postprocessor(outputs, target_sizes)
        # DETR classes start at 0
        labels += 1
        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
        # apply nms
        if self.nms_threshold is not None:
            for result in results:
                keep = torchvision.ops.batched_nms(result["boxes"], result["scores"], result["labels"], self.nms_threshold)
                for k in result.keys():
                    result[k] = result[k][keep]
        return results

    def forward(self, x, ids=None, bboxes=None, labels=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor])
        """
        x: one image of dimensions [batch size, channel count, height, width]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        sample = x
        image_sizes = torch.zeros([len(l_x), 2]).to(x.device)
        i = 0
        for x in l_x:
            image_sizes[i][0] = x.shape[2]
            image_sizes[i][1] = x.shape[1]
            i += 1

        # default placeholders
        dd_outputs = [{"dummy": torch.zeros((0,))}]
        detr_targets: List[Dict[str, torch.Tensor]] = []
        detr_indices = [torch.zeros((0,))]

        # get targets
        if self.training:
            detr_targets = self.dd_targets_to_detr_targets(
                ids, bboxes, labels, image_sizes
            )

        # forward with the targets
        detr_outputs = self.model(sample, detr_targets)
        assert "last_outputs" in detr_outputs
        last_outputs = detr_outputs["last_outputs"][0]

        if self.training:
            with torch.no_grad():
                # concatenate all the indices, will be split later after DD solves them
                detr_indices = []
                detr_indices += self.criterion.matcher(last_outputs, detr_targets)
                if "aux_outputs" in detr_outputs:
                    for aux_outputs in detr_outputs["aux_outputs"]:
                        detr_indices += self.criterion.matcher(
                            aux_outputs, detr_targets
                        )
                if "enc_aux_outputs" in detr_outputs:
                    for aux_outputs in detr_outputs["enc_aux_outputs"]:
                        detr_indices += self.criterion.matcher(
                            aux_outputs, detr_targets
                        )
        else:
            with torch.no_grad():
                # converting detr to torchvision detection format
                dd_outputs = self.detr_outputs_to_dd_outputs(last_outputs, image_sizes)

        return dd_outputs, detr_outputs, detr_targets, detr_indices

    @torch.jit.export
    def loss(self, outputs, targets, indices):
        # type: (Dict[str, List[Dict[str, Tensor]]], List[Dict[str, Tensor]], List[Tensor])
        # convert List[Tensor] of 2D tensors indices to List[Tuple[Tensor, Tensor]] as expected by DETR criterion
        indices = [(x[0], x[1]) for x in indices]
        # split the indices by targets
        indices = [
            indices[i : i + len(targets)] for i in range(0, len(indices), len(targets))
        ]
        # losses are already weighted by criterion
        losses = self.criterion(outputs, targets, indices)
        # make sure we consumed all the indices
        assert len(indices) == 0
        # DD expects a total_loss key as the model loss
        losses["total_loss"] = torch.stack(list(losses.values())).sum()
        return losses


# map model names to their config files
configs = {
    # base models
    "rtdetrv2_s": "rtdetrv2_r18vd_120e_coco.yml",
    "rtdetrv2_m": "rtdetrv2_r50vd_m_7x_coco.yml",
    "rtdetrv2_l": "rtdetrv2_r50vd_6x_coco.yml",
    "rtdetrv2_x": "rtdetrv2_r101vd_6x_coco.yml",
    # discrete sampling
    "rtdetrv2_s_dsp": "rtdetrv2_r18vd_dsp_3x_coco.yml",
    "rtdetrv2_m_dsp": "rtdetrv2_r50vd_m_dsp_3x_coco.yml",
    "rtdetrv2_l_dsp": "rtdetrv2_r50vd_dsp_1x_coco.yml",
}

parser = argparse.ArgumentParser(description="Trace RT-DETR model")
parser.add_argument("model", type=str, help="Model to export", choices=configs.keys())
parser.add_argument(
    "--path-to-rtdetrv2", help="path to rtdetrv2 repository", required=True
)
parser.add_argument("--model-in-file", help="path to model .pth file")
parser.add_argument(
    "-o",
    "--output-dir",
    default=".",
    type=str,
    help="Output directory for traced models",
)
parser.add_argument(
    "--num_classes", type=int, default=81, help="Number of classes of the model"
)
parser.add_argument(
    "--num_queries", default=300, type=int, help="Number of query slots"
)
parser.add_argument(
    "--img_width",
    type=int,
    default=640,
    help="Width of eval_spatial_size",
)
parser.add_argument(
    "--img_height",
    type=int,
    default=640,
    help="Height of eval_spatial_size",
)
parser.add_argument(
    "--nms_threshold",
    type=float,
    default=None,
    help="Enable NMS with the specified IoU threshold"
)
args = parser.parse_args()

# DETR already reserves a no-object class
args.num_classes -= 1

# load model
sys.path.append(args.path_to_rtdetrv2)
from src.core import YAMLConfig

# from https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/tools/export_onnx.py
config = args.path_to_rtdetrv2 + "/configs/rtdetrv2/" + configs[args.model]
cfg = YAMLConfig(
    config,
    resume=args.model_in_file,
    eval_spatial_size=[args.img_height, args.img_width],
    num_classes=args.num_classes,
    PResNet={"pretrained": args.model_in_file is not None},
    RTDETRTransformerv2={
        "num_queries": args.num_queries,
        "num_denoising": math.ceil(args.num_queries / 3),
    },
    RTDETRPostProcessor={"num_top_queries": args.num_queries},
)

# load checkpoint
if args.model_in_file:
    checkpoint = torch.load(args.model_in_file, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    # handle keys that moved due to tracing
    state = {
        k.replace("decoder.query_pos_head", "decoder.decoder.query_pos_head")
        .replace("decoder.dec_bbox_head", "decoder.decoder.bbox_head")
        .replace("decoder.dec_score_head", "decoder.decoder.score_head"): v
        for k, v in state.items()
    }

    # remove keys incompatible with num_classes
    if args.num_classes != 80:
        state = {
            k: v
            for k, v in state.items()
            if not any(
                k.startswith(x)
                for x in [
                    "decoder.denoising_class_embed",
                    "decoder.enc_score_head",
                    "decoder.decoder.score_head",
                ]
            )
        }

    # remove keys incompatible with resolution
    if args.img_width != 640 or args.img_height != 640:
        state = {
            k: v
            for k, v in state.items()
            if not any(
                k.startswith(x)
                for x in [
                    "decoder.anchors",
                    "decoder.valid_mask",
                ]
            )
        }

    cfg.model.load_state_dict(state, strict=False)

# wrap model
model = WrappedRTDETR(cfg, args.nms_threshold)
model.cuda()
model.eval()
filename = os.path.join(
    args.output_dir,
    args.model
    + "_"
    + str(args.img_width)
    + "x"
    + str(args.img_height)
    + "_cls"
    + str(args.num_classes + 1)
    + "_queries"
    + str(args.num_queries)
    + ("_nms" + str(args.nms_threshold) if args.nms_threshold else "")
    + ("_pretrained" if args.model_in_file else "")
    + ".pt",
)
print("Attempting jit export...")
model_jit = torch.jit.script(model)
model_jit.save(filename)
print("jit detr export successful")
