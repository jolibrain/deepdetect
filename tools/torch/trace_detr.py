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
        
    @torch.no_grad()
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

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox).cpu()
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class WrappedDETR(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pp = PostProcess()
        
    def forward(self, x):
        """
        x: one image of dimensions [batch size, channel count, width, height]
        """
        l_x = [x[i] for i in range(x.shape[0])]
        sample = nested_tensor_from_tensor_list(l_x)
        output = self.model(sample)
        image_sizes = torch.zeros([len(l_x),2]).cpu()
        i = 0
        for x in l_x:
            image_sizes[i][0] = x.shape[1]
            image_sizes[i][1] = x.shape[2]
            i += 1
            
        # converting detr to torchvision detection format
        processed_output = self.pp(output['pred_logits'], output['pred_boxes'], image_sizes)
        return processed_output

parser = argparse.ArgumentParser(description="Trace DETR model")
parser.add_argument('--model-in-file',help='path to model .pth file')
parser.add_argument('--dataset-file',type=str,help='unused',default='coco')
parser.add_argument('--device',default='cuda',help='device used for inference')
parser.add_argument('--path-to-detr',help='path to detr repository',required=True)

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

sys.path.append(args.path_to_detr)
import models
from models import build_model
from util.misc import nested_tensor_from_tensor_list

model_without_ddp, criterion, postprocessors = build_model(args)
model_without_ddp.eval()
checkpoint = torch.load(args.model_in_file,map_location='cpu')
model_without_ddp.load_state_dict(checkpoint['model'])
model_without_ddp = WrappedDETR(model_without_ddp)
model_without_ddp.cuda()

## code for inference / testing
#print('predict on image\n')
#data_transforms = transforms.Compose([transforms.ToTensor()])
#output = model_without_ddp(image_loader(data_transforms, '/home/beniz/bus.jpg'))
#print(output)

print('Attempting jit export...')
model_jit = torch.jit.script(model_without_ddp)
model_jit.save(args.model_in_file.replace('.pth','.pt'))
print('jit detr export successful')
