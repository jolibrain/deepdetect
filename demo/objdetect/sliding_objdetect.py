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
description = 'image classification'
mllib = 'caffe'
mltype = 'supervised'
nclasses = 21
width = height = 300
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
	    # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def predict(imgpath, cf):
    parameters_input = {}
    parameters_mllib = {}
    parameters_output = {'bbox':True, 'confidence_threshold':cf}
    data = [imgpath]
    detect = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    cats = []
    bboxes = []
    predictions = detect['body']['predictions']
    for p in predictions:
        for c in p['classes']:
            cat = c['cat']
            bbox = c['bbox']
            cats.append(cat)
            bboxes.append(bbox)
    return cats, bboxes
            
# main
parser = argparse.ArgumentParser()
parser.add_argument('--img', help='path to image or image folder')
parser.add_argument('--stepsize', type=int, default=320, help='sliding window stepsize, to be set to image input size')
parser.add_argument('--windowsize', type=int, default=640, help='window input size')
parser.add_argument("--model-dir",help="model directory")
parser.add_argument("--nclasses", type=int, default=2, help="number of classes")
parser.add_argument("--cf", type=float, default=0.3, help="bboxes confidence threshold")
parser.add_argument("--output-dir", help="detection maps output directory")
args = parser.parse_args()

# creating ML service
model_repo = args.model_dir
model = {'repository':model_repo}
parameters_input = {'connector':'image','width':args.stepsize,'height':args.stepsize,'bbox':True}
parameters_mllib = {'nclasses':args.nclasses,'gpu':True,'gpuid':0}
parameters_output = {}
try:
    servput = dd.put_service(sname,model,description,mllib,
                             parameters_input,parameters_mllib,parameters_output,mltype)
except: # most likely the service already exists
    pass


if os.path.isfile(args.img):
    images = [args.img]
else:
    images = glob.glob(args.img + '*.*')

for image in images:
    img = cv2.imread(image)
    print(image, ' / shape=',img.shape)

    # output detection map
    detectmap = img.copy()

    # - walk through sliding windows
    i = 0
    for (x, y, window) in sliding_window(img, stepSize=args.stepsize, windowSize=(args.windowsize, args.windowsize)):

        # - if window is smaller than input sizes, fill it up correctly
        windowtmp = window.copy()
        resized = False
        if window.shape[0] != args.stepsize or window.shape[1] != args.stepsize:
            resized = True
            windowfull = np.zeros((args.windowsize, args.windowsize, 3), np.uint8)
            windowfull[0: window.shape[0], 0: window.shape[1]] = window.copy()
            window = windowfull

        # - get the local image window
        windowpath = '/tmp/img'+str(i)+'.png' 
        cv2.imwrite(windowpath, window)

        # - process with DD
        cats, bboxes = predict(windowpath, args.cf)

        # - store the output map
        for bbox in bboxes:
            # translate bbox coordinates
            tr_xmin = int(bbox['xmin']) + x
            tr_xmax = int(bbox['xmax']) + x
            tr_ymin = int(bbox['ymin']) + y
            tr_ymax = int(bbox['ymax']) + y
            cv2.rectangle(detectmap, (tr_xmin,tr_ymax),(tr_xmax,tr_ymin),(255,0,0),2)

        i += 1

    # - save the output map
    cv2.imwrite(args.output_dir + '/' + os.path.basename(image).replace('.png','')+'_detectmap.png', detectmap)
