import os
import sys
import argparse
import torch


class WrappedRTDETR(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.criterion = cfg.criterion

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
        img_h, img_w = target_sizes[0].unbind(0)

        # convert to [xc, yc, w, h] format
        bboxes = self.box_xyxy_to_cxcywh(bboxes)

        # and to relative [0, 1] coordinates
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0).cuda()
        bboxes = bboxes / scale_fct

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
        return results

    def forward(self, x, ids=None, bboxes=None, labels=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor])
        """
        x: one image of dimensions [batch size, channel count, width, height]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        sample = x
        image_sizes = torch.zeros([len(l_x), 2]).cuda()
        i = 0
        for x in l_x:
            image_sizes[i][0] = x.shape[1]
            image_sizes[i][1] = x.shape[2]
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

        if self.training:
            with torch.no_grad():
                # do the match
                detr_indices = self.criterion.matcher(detr_outputs, detr_targets)
        else:
            with torch.no_grad():
                dd_outputs = self.detr_outputs_to_dd_outputs(detr_outputs, image_sizes)
                # converting detr to torchvision detection format
                # dd_outputs = self.pp(detr_outputs['pred_logits'], detr_outputs['pred_boxes'], image_sizes)

        return dd_outputs, detr_outputs, detr_targets, detr_indices

    @torch.jit.export
    def loss(self, outputs, targets, indices):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]], List[Tensor])
        # convert List[Tensor] of 2D tensors indices to List[Tuple[Tensor, Tensor]] as expected by DETR criterion
        indices = [(x[0], x[1]) for x in indices]
        losses = self.criterion(outputs, targets, indices)
        weights = self.criterion.weight_dict
        losses = {k: losses[k] * weights[k] for k in losses.keys()}
        # DD expects a total_loss key as the model loss
        losses["total_loss"] = torch.stack(list(losses.values())).sum()
        return losses


# map model names to their config files
configs = {
    # base models
    "rtdetrv2_s": "rtdetrv2_r18vd_120e_coco.yml",
    "rtdetrv2_m": "rtdetrv2_r50vd_m_7x_coco.yml",
    "rtdetrv2_l": "rtdetrv2_r50vd_6x_coco.yml",
    "rtdetrv2_x": "rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
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
# parser.add_argument(
#    "--num_queries", default=300, type=int, help="Number of query slots"
# )
args = parser.parse_args()

# DETR already reserves a no-object class
args.num_classes -= 1

# load model
sys.path.append(args.path_to_rtdetrv2)
from src.core import YAMLConfig

# TODO handle cfg eval_spatial_size, 640x640 by default
config = args.path_to_rtdetrv2 + "/configs/rtdetrv2/" + configs[args.model]

# from https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/tools/export_onnx.py
cfg = YAMLConfig(
    config,
    resume=args.model_in_file,
    num_classes=args.num_classes,
    PResNet={"pretrained": args.model_in_file is not None},
    # RTDETRTransformerv2={"num_queries": args.num_queries},
    # RTDETRPostProcessor={"num_top_queries": args.num_queries},
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

    cfg.model.load_state_dict(state, strict=False)

# wrap model
model = WrappedRTDETR(cfg)
model.cuda()
model.eval()
filename = os.path.join(
    args.output_dir,
    args.model + "_cls" + str(args.num_classes + 1)
    # + "_queries"
    # + str(args.num_queries)
    + ("_pretrained" if args.model_in_file else "") + ".pt",
)
print("Attempting jit export...")
model_jit = torch.jit.script(model)
model_jit.save(filename)
print("jit detr export successful")
quit()


# TODO: remove? some debug code below
from PIL import Image
from torchvision import transforms


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


## code for inference / testing
print("predict on image\n")
data_transforms = transforms.Compose(
    [transforms.Resize((640, 640)), transforms.ToTensor()]
)
images = image_loader(data_transforms, "/home/royale/dog.jpg").cuda()
size = torch.tensor([[640, 640]])

# targets in DD format
ids = torch.tensor([0]).cuda()
labels = torch.tensor([18]).cuda()
boxes = torch.tensor([[30.0, 20.0, 330.0, 240.0]]).cuda()

from scipy.optimize import linear_sum_assignment

# torch inference
if 0:
    model.eval()
    outputs = model(images, ids, boxes, labels)
    print(outputs)

# torch train
if 0:
    model.train()
    _, detr_outputs, detr_targets, detr_indices = model.forward(
        images, ids, boxes, labels
    )
    with torch.no_grad():
        print("detr_indices", detr_indices)
        print("len detr_indices", len(detr_indices))
        print("detr_index.shape", detr_indices[0].shape)
        detr_indices = [torch.tensor(linear_sum_assignment(i)) for i in detr_indices]
        print(detr_indices)
        print(detr_indices[0].shape)
        print(detr_indices[0].dtype)
        print("linear_sum_assignment", detr_indices)
        # detr_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in detr_indices]
    losses = model.loss(detr_outputs, detr_targets, detr_indices)
    from pprint import pprint

    pprint(losses)
    quit()

# jit inference
if 0:
    model_jit.eval()
    outputs = model_jit(images, ids, boxes, labels)
    print(outputs)

# jit train
if 0:
    model_jit.train()
    _, detr_outputs, detr_targets, detr_indices = model_jit.forward(
        images, ids, boxes, labels
    )
    with torch.no_grad():
        print("detr_indices", detr_indices)
        print("len detr_indices", len(detr_indices))
        print("detr_index.shape", detr_indices[0].shape)
        detr_indices = [torch.tensor(linear_sum_assignment(i)) for i in detr_indices]
        print(detr_indices)
        print(detr_indices[0].shape)
        print(detr_indices[0].dtype)
        print("linear_sum_assignment", detr_indices)
        # detr_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in detr_indices]
    losses = model_jit.loss(detr_outputs, detr_targets, detr_indices)
    from pprint import pprint

    pprint(losses)
