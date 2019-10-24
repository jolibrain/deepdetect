import os, sys, argparse
import numpy as np
from dd_client import DD
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

random.seed(134124)

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="path to image")
parser.add_argument("--nclasses",help="number of classes",type=int,default=150)
parser.add_argument("--width",help="image width",type=int,default=480)
parser.add_argument("--height",help="image height",type=int,default=480)
parser.add_argument("--model-dir",help="model directory")
args = parser.parse_args();

host = 'localhost'
port = 8080
sname = 'segserv'
description = 'image segmentation'
mllib = 'caffe'
mltype = 'unsupervised'
nclasses = args.nclasses
width = args.width
height = args.height
dd = DD(host,port)
dd.set_return_format(dd.RETURN_PYTHON)

def random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.randint(0,255) for i in range(3)]
    return [r, g, b]

# creating ML service
model_repo = args.model_dir
if not model_repo:
    model_repo = os.getcwd() + '/model/'
model = {'repository':model_repo}
parameters_input = {'connector':'image','width':width,'height':height}
parameters_mllib = {'nclasses':nclasses}
parameters_output = {}
try:
    servput = dd.put_service(sname,model,description,mllib,
                             parameters_input,parameters_mllib,parameters_output,mltype)
except: # most likely the service already exists
    pass

# prediction call
parameters_input = {'segmentation':True}
parameters_mllib = {'gpu':True,'gpuid':0}
parameters_output = {}
data = [args.image]
detect = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)

pixels = np.array((map(int,detect['body']['predictions'][0]['vals'])))
imgsize = detect['body']['predictions'][0]['imgsize']

# visual output
label_colours = []
for c in range(nclasses):
    label_colours.append(random_color())
label_colours = np.array(label_colours)

r = pixels.copy()
g = pixels.copy()
b = pixels.copy()
for l in range(0,nclasses):
    r[pixels==l] = label_colours[l,0]
    g[pixels==l] = label_colours[l,1]
    b[pixels==l] = label_colours[l,2]

r = np.reshape(r,(imgsize['height'],imgsize['width']))
g = np.reshape(g,(imgsize['height'],imgsize['width']))
b = np.reshape(b,(imgsize['height'],imgsize['width']))
rgb = np.zeros((imgsize['height'],imgsize['width'],3))
rgb[:,:,0] = r/255.0
rgb[:,:,1] = g/255.0
rgb[:,:,2] = b/255.0

plt.figure()
plt.imshow(rgb,vmin=0,vmax=1)
plt.show()
