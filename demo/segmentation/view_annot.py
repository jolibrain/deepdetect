import os, sys, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

random.seed(134124)

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="path to image")
parser.add_argument("--nclasses",help="number of classes",type=int,default=150)
parser.add_argument("--width",help="image width",type=int,default=480)
parser.add_argument("--height",help="image height",type=int,default=480)
args = parser.parse_args();

def random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.randint(0,255) for i in range(3)]
    return [r, g, b]

nclasses = args.nclasses
label_colours = []
for c in range(nclasses):
    label_colours.append(random_color())
label_colours = np.array(label_colours)

height = args.height
width = args.width

pixels = cv2.imread(args.image,cv2.CV_LOAD_IMAGE_GRAYSCALE)

r = pixels.copy()
g = pixels.copy()
b = pixels.copy()
for l in range(0,nclasses):
    r[pixels==l] = label_colours[l,0]
    g[pixels==l] = label_colours[l,1]
    b[pixels==l] = label_colours[l,2]

r = np.reshape(r,(height,width))
g = np.reshape(g,(height,width))
b = np.reshape(b,(height,width))
rgb = np.zeros((height,width,3))
rgb[:,:,0] = r/255.0
rgb[:,:,1] = g/255.0
rgb[:,:,2] = b/255.0

plt.figure()
plt.imshow(rgb,vmin=0,vmax=1)
plt.show()
