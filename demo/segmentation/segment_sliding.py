import argparse
import cv2
import glob
import os
import random
import numpy as np
import sys

from dd_client import DD

host = 'localhost'
sname = 'imgserv'
description = 'image segmentation'
mllib = 'torch'
mltype = 'supervised'
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

def random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.randint(0,255) for i in range(3)]
    return [r, g, b]

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
	    # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def predict(imgpath):
    parameters_input = {'scale':0.0039,'rgb':False}
    parameters_mllib = {'segmentation':True}
    parameters_output = {}
    data = [imgpath]
    detect = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    pixels = np.array(detect['body']['predictions'][0]['vals']).astype(int)
    imgsize = detect['body']['predictions'][0]['imgsize']
    return pixels, imgsize
            
# main
parser = argparse.ArgumentParser()
parser.add_argument('--img', help='path to image')
parser.add_argument('--stepsize', type=int, default=512, help='sliding window stepsize, to be set to image input size')
parser.add_argument("--model-dir",help="model directory")
parser.add_argument("--nclasses", type=int, default=3, help="number of classes")
args = parser.parse_args()

# creating ML service
model_repo = args.model_dir
model = {'repository':model_repo}
parameters_input = {'connector':'image','width':args.stepsize,'height':args.stepsize, 'scale': 0.0039}
parameters_mllib = {'nclasses':args.nclasses,'segmentation':True,'gpu':True,'gpuid':0}
parameters_output = {}
try:
    servput = dd.put_service(sname,model,description,mllib,
                             parameters_input,parameters_mllib,parameters_output,mltype)
except: # most likely the service already exists
    pass

img = cv2.imread(args.img)
print('image shape=',img.shape)

# visual output
label_colours = []
for c in range(args.nclasses):
    label_colours.append(random_color())
label_colours = np.array(label_colours)

# output segmentation map
segmap = np.zeros((img.shape[0],img.shape[1],3), np.uint8)

# - walk through sliding windows
i = 0
for (x, y, window) in sliding_window(img, stepSize=args.stepsize, windowSize=(args.stepsize, args.stepsize)):
    #print('window shape=',window.shape)
    
    # - if window is smaller than input sizes, fill it up correctly
    windowtmp = window.copy()
    resized = False
    if window.shape[0] != args.stepsize or window.shape[1] != args.stepsize:
        resized = True
        windowfull = np.zeros((args.stepsize, args.stepsize, 3), np.uint8)
        windowfull[0: window.shape[0], 0: window.shape[1]] = window.copy()
        window = windowfull
    
    # - get the local image window
    windowpath = '/tmp/seg/img'+str(i)+'.jpg' 
    cv2.imwrite(windowpath, window)

    # - process with DD
    pixels, imgsize = predict(windowpath)
        
    # - store the output map
    #print(label_colours)
    r = pixels.copy()
    g = pixels.copy()
    b = pixels.copy()
    for l in range(0,args.nclasses):
        r[pixels==l] = label_colours[l,0]
        g[pixels==l] = label_colours[l,1]
        b[pixels==l] = label_colours[l,2]
    r = np.reshape(r,(imgsize['height'],imgsize['width']))
    g = np.reshape(g,(imgsize['height'],imgsize['width']))
    b = np.reshape(b,(imgsize['height'],imgsize['width']))
    rgb = np.zeros((imgsize['height'],imgsize['width'],3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    # - combine the output maps
    if resized:
        rgb = rgb[0:windowtmp.shape[0],0:windowtmp.shape[1]]
    segmap[y: y + window.shape[1], x: x + window.shape[0]] = rgb
    
    i += 1

# - save the output map
cv2.imwrite('outseg.jpg', segmap)
