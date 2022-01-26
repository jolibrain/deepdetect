import os, sys, argparse
from dd_client import DD
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="path to image")
parser.add_argument("--port", help="DeepDetect port", type=int, default=8080)
parser.add_argument("--confidence-threshold",help="keep detections with confidence above threshold",type=float,default=0.1)
parser.add_argument("--save-path", help="Where to save resulting image")
args = parser.parse_args()

host = 'localhost'
sname = 'imgserv'
description = 'image classification'
mllib = 'caffe'
mltype = 'supervised'
nclasses = 21
width = height = 300
dd = DD(host, port=args.port)
dd.set_return_format(dd.RETURN_PYTHON)

# creating ML service
model_repo = os.getcwd() + '/model'
model = {'repository':model_repo}
parameters_input = {'connector':'image','width':width,'height':height}
parameters_mllib = {'nclasses':nclasses}
parameters_output = {}
dd.put_service(sname,model,description,mllib,
               parameters_input,parameters_mllib,parameters_output,mltype)

# chain call
calls = []

parameters_input = {"keep_orig":True}
parameters_mllib = {'gpu':True}
parameters_output = {'bbox':True, 'confidence_threshold': args.confidence_threshold}
data = [args.image]
calls.append(dd.make_call(sname, data, parameters_input, parameters_mllib, parameters_output))

parameters_action = {"output_images":True, "write_prob": True}
# parameters_action["save_path"] = os.getcwd()
# parameters_action["save_img"] = True
calls.append(dd.make_action("draw_bbox", parameters_action, "img_bbox"))

detect = dd.post_chain("chain_ddetect",calls)
# print(detect)
if detect['status']['code'] != 200:
    print('error',detect['status']['code'])
    sys.exit()

predictions = detect['body']['predictions']
for p in predictions:
    # get orig img dimensions
    orig_img = cv2.imread(p['uri'])
    width, height, _ = orig_img.shape

    img = np.array(p["img_bbox"]["vals"])
    img = img.reshape((width, height, 3))

    if args.save_path:
        cv2.imwrite(args.save_path, img)
    cv2.imshow('img',img)
    k = cv2.waitKey(0)
