#!/usr/bin/python3

import sys
import os
import argparse
import logging
import json
import torch

import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(
        description="""
General usage (examples):
- Export yolox model for DeepDetect training:

    python3 trace_yolox.py -v yolox[-s|-m|-l|...] --yolox_path [YOLOX_PATH] --output_dir [OUTPUT_DIR] --num_classes 3 --img_width 512 --img_height 512

- Export dd-trained yolox model to onnx for trt inference:

    python3 trace_yolox.py -v yolox[-s|-m|-l|...] --yolox_path [YOLOX_PATH] --output_dir [OUTPUT_DIR] --from_repo [DD_REPO] --to_onnx
""", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=str, help="Model to export")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="", help="Output directory"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )
    parser.add_argument(
        "--from_repo", type=str, help="Trace from a deepdetect model repository"
    )
    parser.add_argument("--weights", type=str, help="yolo-x weights file (.pth or .pt)")
    parser.add_argument(
        "--backbone_weights",
        type=str,
        help="yolo-x weights file, but will be applied only to backbone",
    )
    parser.add_argument("--yolox_path", type=str, help="Path of yolo-x repository")
    parser.add_argument(
        "--num_classes", type=int, default=81, help="Number of classes of the model"
    )
    parser.add_argument("--gpu", type=int, help="GPU id to run on GPU")
    parser.add_argument("--to_onnx", action="store_true", help="Export model to onnx")
    parser.add_argument(
        "--use_wrapper",
        action="store_true",
        help="In case of onnx export, if this option is present, the model will be wrapped so that its output match dede expectations",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="When exporting to onnx, specify maximum returned prediction count",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="When exporting to onnx, batch size of model",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=640,
        help="Width of the image when exporting with fixed image size",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=640,
        help="Height of the image when exporting with fixed image size",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    device = "cpu"
    if args.gpu:
        device = "cuda:%d" % args.gpu

    if args.from_repo:
        fill_args_from_repo(args.from_repo, args)

    if not args.model:
        raise ValueError("No model specified")

    # get yolox model
    sys.path.insert(0, args.yolox_path)
    import yolox
    from yolox.exp import get_exp
    from yolox.utils import get_model_info, postprocess, replace_module

    from yolox.models.network_blocks import SiLU

    exp = get_exp(None, args.model)
    # dede assumes a background class absent from yolox
    exp.num_classes = args.num_classes - 1
    logging.info("num_classes == %d" % args.num_classes)

    model = exp.get_model()
    model.eval()
    model.head.decode_in_inference = True

    if args.weights:
        logging.info("Load weights from %s" % args.weights)

        def load_yolox_weights():
            try:
                # state_dict
                weights = torch.load(args.weights)["model"]
            except:
                # torchscript
                logging.info("Detected torchscript weights")
                weights = torch.jit.load(args.weights).state_dict()
                weights = {k[6:]: w for k, w in weights.items()}  # skip "model." prefix

            model.load_state_dict(weights, strict=True)

        try:
            load_yolox_weights()
        except:
            # Legacy model
            exp.num_classes = args.num_classes

            exp.model = None
            model = exp.get_model()
            model.eval()
            model.head.decode_in_inference = True

            load_yolox_weights()
            logging.info("Detected yolox trained with a background class")

    elif args.backbone_weights:
        logging.info("Load weights from %s" % args.backbone_weights)

        weights = torch.load(args.backbone_weights)["model"]
        weights = {k: w for k, w in weights.items() if "backbone" in k}

        model.load_state_dict(weights, strict=False)

    logging.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    filename = os.path.join(args.output_dir, args.model)

    if args.to_onnx:
        model = replace_module(model, nn.SiLU, SiLU)

        model = YoloXWrapper_TRT(
            model, topk=args.top_k, raw_output=not args.use_wrapper
        )
        model.to(device)
        model.eval()

        filename += ".onnx"
        example = get_image_input(args.batch_size, args.img_width, args.img_height)
        # XXX: dynamic batch size not supported with wrapper
        # XXX: dynamic batch size not yet supported in dede as well
        dynamic_axes = None  # {"input": {0: "batch"}} if not args.use_wrapper else None
        torch.onnx.export(
            model,
            example,
            filename,
            export_params=True,
            verbose=args.verbose,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["detection_out", "keep_count"],
            dynamic_axes=dynamic_axes,
        )
    else:
        # wrap model
        model = YoloXWrapper(model, exp.num_classes, postprocess)
        model.to(device)
        model.eval()

        filename += "_cls" + str(args.num_classes) + ".pt"
        script_module = torch.jit.script(model)
        logging.info("Save jit model at %s" % filename)
        script_module.save(filename)


# ====


class YoloXWrapper(torch.nn.Module):
    def __init__(self, model, num_classes, postprocess):
        super(YoloXWrapper, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.postprocess = postprocess
        self.nms_threshold = 0.45

        self.model.head.reg_weight = 5.0
        self.model.head.use_l1 = True

    def convert_targs(self, bboxes):
        """
        Converts bboxes from box corners (dd format) to center + size
        (yolox format)
        """
        yolox_boxes = bboxes.new_zeros(bboxes.shape)
        yolox_boxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        yolox_boxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        yolox_boxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        yolox_boxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return yolox_boxes

    def forward(self, x, ids=None, bboxes=None, labels=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Dict[str,Tensor], List[Dict[str, Tensor]]]

        placeholder = {
            "boxes": torch.zeros((0, 4), device=x.device),
            "scores": torch.zeros((0,), device=x.device),
            "labels": torch.zeros((0,), device=x.device, dtype=torch.int64),
        }

        if self.training:
            assert ids is not None
            assert bboxes is not None
            assert labels is not None

            l_targs = []
            stop = 0
            max_count = 0

            for i in range(x.shape[0]):
                start = stop

                while stop < ids.shape[0] and ids[stop] == i:
                    stop += 1

                bboxes[start:stop]
                targ = torch.cat(
                    (
                        labels[start:stop].unsqueeze(1),
                        self.convert_targs(bboxes[start:stop]),
                    ),
                    dim=1,
                )
                # dd uses 0 as background class, not YOLOX
                targ = targ - 1
                l_targs.append(targ)
                max_count = max(max_count, targ.shape[0])

            l_targs = [
                F.pad(targ, (0, 0, 0, max_count - targ.shape[0])) for targ in l_targs
            ]
            targs = torch.stack(l_targs, dim=0)
            output, losses = self.model(x, targs)
            preds = [placeholder]
        else:
            losses = {}
            with torch.no_grad():
                output = self.model(x)[0]
                preds_list = self.postprocess(
                    output, self.num_classes, 0.01, self.nms_threshold
                )

            # Must initialize list with something in it so that torchscript can deduce type
            preds = [placeholder][1:]

            for pred in preds_list:
                if pred.shape[0] == 0:
                    preds.append(placeholder)
                else:
                    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                    preds.append(
                        {
                            "boxes": pred[:, :4],
                            "scores": pred[:, 4] * pred[:, 5],
                            # dd uses 0 as background class, not YOLOX
                            "labels": pred[:, 6].to(torch.int64) + 1,
                        }
                    )

        return losses, preds


class YoloXWrapper_TRT(torch.nn.Module):
    def __init__(self, model, topk=200, raw_output=False):
        super(YoloXWrapper_TRT, self).__init__()
        self.model = model
        self.topk = topk
        self.raw_output = raw_output

    def to_xyxy(self, boxes):
        xyxy_boxes = boxes.new_zeros(boxes.shape)
        xyxy_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        xyxy_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        xyxy_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        xyxy_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
        return xyxy_boxes

    def forward(self, x):
        # xmin, ymin, xmax, ymax, objectness, conf cls1, conf cl2...
        output = self.model(x)[0]

        if self.raw_output:
            return output, torch.zeros(output.shape[0])

        box_count = output.shape[1]
        cls_scores, cls_pred = output[:, :, 5:].max(dim=2, keepdim=True)
        batch_ids = (
            torch.arange(output.shape[0], device=x.device)
            .view(-1, 1)
            .repeat(1, output.shape[1])
            .unsqueeze(2)
        )

        # to dede format: batch id, class id, confidence, xmin, ymin, xmax, ymax
        scores = cls_scores * output[:, :, 4].unsqueeze(2)
        output = torch.cat(
            (batch_ids.to(x.dtype), cls_pred.to(x.dtype), scores, output[:, :, :4]),
            dim=2,
        )

        # Return sorted topk values
        sort_indices = scores.topk(self.topk, dim=1).indices.reshape(-1)
        sort_indices += (
            batch_ids[:, : self.topk, :].reshape(-1).to(torch.int64) * box_count
        )
        output = output.reshape(-1, 7)[sort_indices.squeeze(0)].contiguous()

        # convert bboxes to dd format
        output[:, 3:7] = self.to_xyxy(output[:, 3:7])
        output[:, 3] /= x.shape[2] - 1
        output[:, 4] /= x.shape[3] - 1
        output[:, 5] /= x.shape[2] - 1
        output[:, 6] /= x.shape[3] - 1

        # detection_out, keep_count
        return output, output[:, 0]


from PIL import Image


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


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


def test_model(model, script_module):
    # loading test image
    data_transforms = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor()]
    )
    img = image_loader(data_transforms, "bus.jpg")
    img = img.to(device).float() * 255.0

    # inference on
    output = model(img)[0]
    u = int(torch.argmax(output[0, :, 4]).item())
    print("Original model output:", u, output[0, u])

    # inps = get_image_input(2, 640, 640)
    # model(get_image_input(2, 640, 640))
    script_module.eval()
    outputs = script_module(img).to(device)
    # outputs = script_module(get_image_input(2, 640, 640).to(device))
    print("Eval call successful! Outputs:", outputs)

    # torch.autograd.set_detect_anomaly(True)
    script_module.train()
    outputs = script_module(*[t.to(device) for t in get_detection_input(2, 640, 640)])
    outputs[0].backward()
    print("Train call successful! Outputs:", outputs)


def fill_args_from_repo(repo_path, args):
    config_fname = os.path.join(repo_path, "config.json")
    with open(config_fname) as config_file:
        config_json = json.load(config_file)
        args.img_width = config_json["parameters"]["input"]["width"]
        args.img_height = config_json["parameters"]["input"]["height"]
        args.num_classes = config_json["parameters"]["mllib"]["nclasses"]

    best_model_fname = os.path.join(repo_path, "best_model.txt")
    with open(best_model_fname) as best_model_file:
        for line in best_model_file.readlines():
            if line.startswith("iteration:"):
                it = int(line.split(":")[1].strip())
        args.weights = os.path.join(repo_path, "checkpoint-%d.pt" % it)

    # Try to deduce template if not present
    if not args.model or args.model == "auto":
        suffixes = {"_m": "yolox-m", "-m": "yolox-m", "_s": "yolox-s", "-s": "yolox-s"}
        repo_name = os.path.basename(repo_path.rstrip("/"))
        model_found = False

        for suffix in suffixes:
            if repo_name.endswith(suffix):
                args.model = suffixes[suffix]
                model_found = True
                logging.info("Deduced model %s with suffix %s" % (args.model, suffix))

        if not model_found:
            raise RuntimeError("Could not deduce the model from repository name %s" % repo_name)


if __name__ == "__main__":
    main()
