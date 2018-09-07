#!/usr/bin/env python

import os
import json
import argparse
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def pred_to_fig(pred, alpha=0.9, color_min=0.3, color_max=0.7):
    name = os.path.split(pred['uri'])[-1]
    print('Applying masks on "{}"'.format(name))
    img = np.array(Image.open(pred['uri']).convert('RGBA'))
    fig, ax = plt.subplots(1)
    ax.set_title(name)
    ax.imshow(img)
    for item in pred['classes']:

        #Fetch
        xmin = int(item['bbox']['xmin'])
        ymin = int(item['bbox']['ymin'])
        width = item['mask']['width']
        height = item['mask']['height']

        #Set to a random color
        mask = np.array(item['mask']['data']).astype(float) * 255
        mask = np.stack((mask.reshape(height, width),) * 4, -1)
        mask[...,-1] *= alpha
        mask[...,:-1] *= np.random.uniform(color_min, color_max, 3)

        #Plot
        buff = np.zeros(img.shape, dtype='uint8')
        buff[ymin:ymin+height, xmin:xmin+width] = mask
        ax.imshow(buff)
        ax.text(xmin, ymin, '{} {:.2f}'.format(item['cat'], item['prob']))

    plt.figure(fig.number)

def preds_to_pdf(data, path):
    with PdfPages(path) as pdf:
        for pred in data['body']['predictions']:
            pred_to_fig(pred)
            pdf.savefig()

def get_preds(host, port, service, thresh, images):
    url = 'http://{}:{}/predict'.format(host, port)
    print('Posting on "{}"'.format(url))
    return requests.post(url, json = {
        'service': service,
        'parameters': {
            'output': {
                'mask': True,
                'best': 1,
                'confidence_threshold': thresh
            }
        },
        'data': images
    }).json()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument('--port', default=8080, type=int)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--pdf', required=True, type=str)
    parser.add_argument('--service', required=True, type=str)
    parser.add_argument('image', type=str, nargs='+')
    return parser.parse_args()

def main():
    args = get_args()
    preds = get_preds(args.host, args.port, args.service, args.threshold, args.image)
    preds_to_pdf(preds, args.pdf)
    return 0

if __name__ == '__main__':
    exit(main())
