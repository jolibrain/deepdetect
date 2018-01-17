import os, sys, argparse
from os import listdir
from os.path import isfile, join
from os import walk
from dd_client import DD
from annoy import AnnoyIndex
import shelve
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--index",help="repository of images to be indexed")
parser.add_argument("--index-batch-size",type=int,help="size of image batch when indexing",default=1)
parser.add_argument("--search",help="image input file for similarity search")
parser.add_argument("--search-size",help="number of nearest neighbors",type=int,default=10)
parser.add_argument("--confidence-threshold",help="confidence threshold on bounding boxes",type=float,default=0.01)
parser.add_argument("--nclasses",help="number of classes in the model",type=int,default=21)
parser.add_argument("--model-dir",help="model directory",default="model")
args = parser.parse_args()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def image_resize(imgfile,width):
    imgquery = cv2.imread(imgfile)
    r = width / imgquery.shape[1]
    dim = (int(width), int(imgquery.shape[0] * r))
    small = cv2.resize(imgquery,dim)
    return small

host = 'localhost'
sname = 'imageserv'
description = 'image classification'
mllib = 'caffe'
mltype = 'supervised'
extract_layer = 'rois'
nclasses = args.nclasses
layer_size = 512 # auto anyways
width = height = 300
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)
ntrees = 1000
metric = 'angular'  # or 'euclidean'

# creating ML service
model_repo = os.getcwd() + '/' + args.model_dir
model = {'repository':model_repo,'templates':'../templates/caffe/'}
parameters_input = {'connector':'image','width':width,'height':height}
parameters_mllib = {'nclasses':nclasses}
parameters_output = {}
try:
    dd.put_service(sname,model,description,mllib,
                   parameters_input,parameters_mllib,parameters_output,mltype)
except:
    pass

# reset call params
parameters_input = {}
parameters_mllib = {'gpu':True}
parameters_output = {'rois':'rois','confidence_threshold':args.confidence_threshold,'best':1}

if args.index:
    try:
        os.remove('data.bin')
    except:
        pass
    s = shelve.open('data.bin')

    # list files in image repository
    c = 0
    d = 1
    onlyfiles = []
    for (dirpath, dirnames, filenames) in walk(args.index):
        nfilenames = []
        for f in filenames:
            nfilenames.append(dirpath + '/' + f)
        onlyfiles.extend(nfilenames)
    for x in batch(onlyfiles,args.index_batch_size):
        classif = dd.post_predict(sname,x,parameters_input,parameters_mllib,parameters_output)

        for p in classif['body']['predictions']:
            uri =  p['uri']
            rois = p['rois']
            sys.stdout.write('\rIndexing image '+str(d)+'/'+str(len(onlyfiles)) + ' : ' + str(len(rois)) + ' rois  total:' + str(c) + '   ')
            sys.stdout.flush()

            for roi in rois:
                bbox = roi['bbox']
                cat = roi['cat']
                prob = roi['prob']
                vals = roi['vals']
                if c == 0:
                    layer_size = len(vals)
                    s['layer_size'] = layer_size
                    t = AnnoyIndex(layer_size,metric) # prepare index
                t.add_item(c,vals)
                s[str(c)] = {'uri':uri, 'bbox' : bbox, 'cat' : cat, 'prob' : prob}
                c = c + 1
            d = d + 1
        #if c >= 10000:
        #    break
    print 'building index...\n'
    print 'layer_size=',layer_size
    t.build(ntrees)
    t.save('index.ann')
    s.close()

if args.search:
    s = shelve.open('data.bin')
    u = AnnoyIndex(s['layer_size'],metric)
    u.load('index.ann')
    data = [args.search]
    classif = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    # search for every roi
    res = classif['body']['predictions'][0]['rois']
    print('number of ROI in query: ' + str(len(res)))
    for roi in res:
        near = u.get_nns_by_vector(roi['vals'],args.search_size,include_distances=True)
        near_data = []
        near_distance = []
        for n in near[1]:
            near_distance.append(n)
        print('distances: ')
        print(near_distance)
        for n in near[0]:
            near_data.append(s[str(n)])
        # print query bbox
        img = cv2.imread(args.search)
        bbox = roi['bbox']
        cat = roi['cat']
        cv2.rectangle(img, (int(bbox['xmin']),int(bbox['ymax'])),(int(bbox['xmax']),int(bbox['ymin'])),(255,0,0),2)

        cv2.putText(img,cat,(int(bbox['xmin']),int(bbox['ymax'])),cv2.FONT_HERSHEY_PLAIN,1,255)
        cv2.imshow('query',img)
        cv2.waitKey(0)
        for n in near_data:
            resimg = cv2.imread(n['uri'])
            bbox = n['bbox']
            cat = n['cat']
            cv2.rectangle(resimg, (int(bbox['xmin']),int(bbox['ymax'])),(int(bbox['xmax']),int(bbox['ymin'])),(255,0,0),2)

            cv2.putText(resimg,cat,(int(bbox['xmin']),int(bbox['ymax'])),cv2.FONT_HERSHEY_PLAIN,1,255)
            cv2.imshow('res',resimg)
            cv2.waitKey(0)

dd.delete_service(sname,clear='')
