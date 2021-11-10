#!/usr/bin/python3

import sys
import os
import argparse
import logging
import torch

import torchvision
import torch.nn.functional as F
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model", type=str, help="Model to export")
    parser.add_argument("-o", "--output_dir", type=str, default="", help="Output directory")
    parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")
    parser.add_argument('--weights', type=str, help="yolo-x weights file (.pth)")
    parser.add_argument('--yolox_path', type=str, help="Path of yolo-x repository")
    parser.add_argument('--num_classes', type=int, default=80, help="Number of classes of the model")
    parser.add_argument('--gpu', type=int, help="GPU id to run on GPU")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    device = "cpu"
    if args.gpu:
        device = "cuda:%d" % args.gpu


    # get yolox model
    sys.path.insert(0, args.yolox_path)
    import yolox
    from yolox.exp import get_exp
    from yolox.utils import get_model_info, postprocess

    exp = get_exp(None, args.model)
    exp.num_classes = args.num_classes

    model = exp.get_model()
    model.eval()
    model.head.decode_in_inference = True

    if args.weights:
        logging.info("Load weights from %s" % args.weights)
        try:
            # state_dict
            weights = torch.load(args.weights)["model"]
            weights = {k : w for k, w in weights.items() if "backbone" in k}
        except:
            # torchscript
            logging.info("Detected torchscript weights")
            weights = torch.jit.load(args.weights).state_dict()
            weights = {k[6:] : w for k, w in weights.items()} # skip "model."

        model.load_state_dict(weights, strict=False)

    logging.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # wrap model
    model = YoloXWrapper(model, args.num_classes, postprocess)
    model.to(device)
    model.eval()

    filename = os.path.join(args.output_dir, args.model)

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

    def forward(self, x, ids = None, bboxes = None, labels = None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, List[Dict[str, Tensor]]]

        placeholder = {
            "boxes": torch.zeros((0,4), device=x.device),
            "scores": torch.zeros((0,), device=x.device),
            "labels": torch.zeros((0,), device=x.device, dtype=torch.int64)
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
                targ = torch.cat((
                    labels[start:stop].unsqueeze(1),
                    self.convert_targs(bboxes[start:stop])
                    ), dim=1)
                l_targs.append(targ)
                max_count = max(max_count, targ.shape[0])

            l_targs = [F.pad(targ, (0, 0, 0, max_count - targ.shape[0])) for targ in l_targs]
            targs = torch.stack(l_targs, dim=0)

            output, losses = self.model(x, targs)
            loss = losses["total_loss"]
            preds = [placeholder]
        else:
            loss = torch.zeros((1,), device=x.device, dtype=x.dtype)
            with torch.no_grad():
                output = self.model(x)[0]
                preds_list = self.postprocess(output, self.num_classes, 0.01, self.nms_threshold)

            # Must initialize list with something in it so that torchscript can deduce type
            preds = [placeholder][1:]

            for pred in preds_list:
                if pred.shape[0] == 0:
                    preds.append(placeholder)
                else:
                    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                    preds.append({
                        "boxes": pred[:,:4],
                        "scores": pred[:,4]*pred[:,5],
                        "labels": pred[:,6].to(torch.int64)
                    })

        return loss, preds

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
    data_transforms = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    img = image_loader(data_transforms, 'bus.jpg')
    img = img.to(device).float() * 255.

    # inference on 
    output = model(img)[0]
    u = int(torch.argmax(output[0,:,4]).item())
    print("Original model output:", u, output[0,u])

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


if __name__ == "__main__":
    main()
