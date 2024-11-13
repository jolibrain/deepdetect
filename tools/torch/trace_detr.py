import sys
import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, List

#from PIL import Image

#def image_loader(loader, image_name):
#    image = Image.open(image_name)
#    image = loader(image).float()
#    image = torch.tensor(image, requires_grad=False)
#    image = image.unsqueeze(0)
#    return image

class PostProcess(torch.nn.Module):
    """ This module converts the model's output into the format expected by DD"""
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
        
    def forward(self, out_logits, out_bbox, target_sizes):
        """ Perform the computation
        Parameters:
            out_logits and out_bbox: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        #assert len(out_logits) == len(target_sizes)
        #assert target_sizes.shape[1] == 2

        prob = self.softmax(out_logits)
        scores, labels = prob[..., :-1].max(-1)
        # DETR classes start at 0
        labels += 1

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox).cpu()
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class PostProcessTrain(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def box_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) * 0.5, (y0 + y1) * 0.5,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def forward(self, detr_outputs, target_sizes, ids, bboxes, labels):
        # type: (Dict[str, Tensor], Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor])
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
        detr_targets : List[Dict[str, torch.Tensor]] = []
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

        with torch.no_grad():
            detr_indices = self.criterion.matcher(detr_outputs, detr_targets)

        return detr_targets, detr_indices

class WrappedDETR(torch.nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.pp = PostProcess()
        self.pp_train = PostProcessTrain(criterion)

    def forward(self, x, ids=None, bboxes=None, labels=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor])
        """
        x: one image of dimensions [batch size, channel count, width, height]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        sample = nested_tensor_from_tensor_list(l_x)

        # default output placeholders
        dd_outputs = [{"dummy": torch.zeros((0, ))}]
        detr_outputs = self.model(sample)
        detr_indices = [torch.zeros((0, ))]
        detr_targets : List[Dict[str, torch.Tensor]] = []

        image_sizes = torch.zeros([len(l_x),2]).cpu()
        i = 0
        for x in l_x:
            image_sizes[i][0] = x.shape[1]
            image_sizes[i][1] = x.shape[2]
            i += 1

        if self.training:
            detr_targets, detr_indices = self.pp_train(detr_outputs, image_sizes, ids, bboxes, labels)
        else:
            with torch.no_grad():
                # converting detr to torchvision detection format
                dd_outputs = self.pp(detr_outputs['pred_logits'], detr_outputs['pred_boxes'], image_sizes)

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
        losses["total_loss"] = torch.stack(losses.values()).sum()
        return losses

parser = argparse.ArgumentParser(description="Trace DETR model")
parser.add_argument('--model-in-file',help='path to model .pth file')
parser.add_argument('--dataset-file',type=str,help='unused',default='coco')
parser.add_argument('--device',default='cuda',help='device used for inference')
parser.add_argument('--path-to-detr',help='path to detr repository',required=True)
parser.add_argument("--num_classes", type=int, default=91 + 1, help="Number of classes of the model")
parser.add_argument('-o', "--output-dir", default=".", type=str, help="Output directory for traced models")

# * Training
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

args = parser.parse_args()
# DETR already reserves a no-object class
args.num_classes -= 1

sys.path.append(args.path_to_detr)
import models
from models import build_model
from util.misc import nested_tensor_from_tensor_list

model_without_ddp, criterion, postprocessors = build_model(args)
model_without_ddp.eval()

if args.model_in_file:
    checkpoint = torch.load(args.model_in_file,map_location='cpu')
    # handle pretrained with different number of classes
    # https://github.com/facebookresearch/detr/issues/9#issuecomment-636391562
    if checkpoint['model']['class_embed.bias'].shape[0] != args.num_classes + 1:
        print('pretrained used different num_classes, removing class_embed')
        del checkpoint['model']['class_embed.weight']
        del checkpoint['model']['class_embed.bias']
    if checkpoint['model']['query_embed.weight'].shape[0] != args.num_queries:
        print('pretrained used different num_queries, removing query_embed')
        del checkpoint['model']['query_embed.weight']
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

model_without_ddp = WrappedDETR(model_without_ddp, criterion)
model_without_ddp.cuda()

## code for inference / testing
#print('predict on image\n')
#data_transforms = transforms.Compose([transforms.ToTensor()])
#output = model_without_ddp(image_loader(data_transforms, '/home/beniz/bus.jpg'))
#print(output)

filename = os.path.join(
    args.output_dir,
    "detr_"
    + args.backbone
    + "_cls"
    + str(args.num_classes + 1)
    + "_queries"
    + str(args.num_queries)
    + ("_pretrained" if args.model_in_file else "")
    + ".pt",
)

print('Attempting jit export...')
model_jit = torch.jit.script(model_without_ddp)
model_jit.save(filename)
print('jit detr export successful')
quit()

# TODO: remove? some debug code below
from scipy.optimize import linear_sum_assignment
from PIL import Image
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image

## code for inference / testing
print('predict on image\n')
data_transforms = transforms.Compose([transforms.ToTensor()])
images = image_loader(data_transforms, '/home/royale/dog.jpg').cuda()

# targets in DD format
ids = torch.tensor([0]).cuda()
labels = torch.tensor([18]).cuda()
boxes = torch.tensor([[30.0, 20.0, 330.0, 240.0]]).cuda()

if 0:
    # check train with bs=2 and various size bboxes
    images = torch.rand((2, 3, 100, 100)).cuda()
    ids = torch.tensor([0, 1, 1]).cuda()
    labels = torch.tensor([10, 11, 11]).cuda()
    boxes = torch.tensor([
        [10., 10., 10., 10.],
        [21., 21., 21., 21.],
        [32., 32., 32., 32.],
    ]).cuda()

if 0:
    # check torch eval
    model_without_ddp.eval()
    outputs = model_without_ddp(image)[0]
    scores, labels, boxes = outputs.values()
    n = scores.argmax()
    print("max", n)
    print("label", labels[n])
    print("bbox", boxes[n])
    quit()

if 0:
    # check torch train
    model_without_ddp.train()
    outputs = model_without_ddp(image, ids, boxes, labels)
    quit()

if 0:
    # check jit eval
    model_jit.eval()
    dd_outputs, _, _, _ = model_jit(images)
    scores, labels, boxes = dd_outputs[0].values()
    n = scores.argmax()
    print("max", n)
    print("label", labels[n])
    print("bbox", boxes[n])
    quit()

# check jit train
model_jit.train()
_, detr_outputs, detr_targets, detr_indices = model_jit.forward(images, ids, boxes, labels)
with torch.no_grad():
    print("detr_indices", detr_indices)
    print("len detr_indices", len(detr_indices))
    print("detr_index.shape", detr_indices[0].shape)
    detr_indices = [torch.tensor(linear_sum_assignment(i)) for i in detr_indices]
    print(detr_indices)
    print(detr_indices[0].shape)
    print(detr_indices[0].dtype)
    print("linear_sum_assignment", detr_indices)
    #detr_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in detr_indices]
#losses = model_jit.criterion(detr_outputs, detr_targets, detr_indices)
losses = model_jit.loss(detr_outputs, detr_targets, detr_indices)
from pprint import pprint
pprint(losses)

# check backward step
if 0:
    loss = sum(x for x in losses.values())
    print(loss)
    layer = list(model_jit.named_parameters())[0][1]
    print("grad before", layer.grad)
    loss.backward()
    print("grad after", layer.grad)
