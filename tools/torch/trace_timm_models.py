import torch
import timm
from timm.models import create_model, list_models
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model-name',help='model name from timm',required=True)
parser.add_argument('--in-chans',help='number of input channels',type=int,default=3)
parser.add_argument('--nclasses',help='number of output classes for the model',type=int,default=2)
parser.add_argument('--pretrained',help='whether to download the model weights if they exist',action='store_true')
parser.add_argument('--no-head', help='whether to not export the model classification head',action='store_true')
args = parser.parse_args()

list_timm_models = list_models()
if not args.model_name in list_timm_models:
    print(list_timm_models)
    print('unknown model ',args.model_name)
    sys.exit()

nclasses = args.nclasses
pooling = 'avg'
if args.no_head:
    nclasses = 0
    pooling = ''
model = create_model(args.model_name, num_classes=nclasses, in_chans=args.in_chans, pretrained=args.pretrained, global_pool=pooling)
jit_model = torch.jit.script(model)
out_model_name = args.model_name
out_model_name += '_inchans' + str(args.in_chans)
if args.no_head:
    out_model_name += '_nohead'
if args.pretrained:
    out_model_name += '_pretrained'
if nclasses > 0:
    out_model_name += '_classes' + str(nclasses)
jit_model.save(out_model_name + ".pt")
