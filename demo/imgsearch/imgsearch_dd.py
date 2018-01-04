import os, sys, argparse
from os import listdir
from os.path import isfile, join
from os import walk
from dd_client import DD
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--index",help="repository of images to be indexed")
parser.add_argument("--index-batch-size",type=int,help="size of image batch when indexing",default=1)
parser.add_argument("--search",help="image input file for similarity search")
parser.add_argument("--search-size",help="number of nearest neighbors",type=int,default=10)
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
sname = 'imgserv'
description = 'image classification'
mllib = 'caffe'
mltype = 'unsupervised'
extract_layer = 'loss3/classifier'
#extract_layer = 'pool5/7x7_s1'
nclasses = 1000
width = height = 224
binarized = False
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

# creating ML service
model_repo = os.getcwd() + '/model'
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
parameters_mllib = {'gpu':True,'extract_layer':extract_layer}
parameters_output = {'binarized':binarized}

if args.index:
    parameters_output['index'] = True
    
    # list files in image repository
    c = 0
    onlyfiles = []
    for (dirpath, dirnames, filenames) in walk(args.index):
        nfilenames = []
        for f in filenames:
            nfilenames.append(dirpath + '/' + f)
        onlyfiles.extend(nfilenames)
    for x in batch(onlyfiles,args.index_batch_size):
        sys.stdout.write('\r'+str(c)+'/'+str(len(onlyfiles)))
        sys.stdout.flush()
        classif = dd.post_predict(sname,x,parameters_input,parameters_mllib,parameters_output)
        for p in classif['body']['predictions']:
            c = c + 1
        if c >= 100:
            break

    # one last dumb predict call to build the index
    print 'building index...\n'
    parameters_output['index'] = False
    parameters_output['build_index']=True
    classif = dd.post_predict(sname,[nfilenames[0]],parameters_input,parameters_mllib,parameters_output)

if args.search:
    parameters_output['search'] = True
    parameters_output['search_nn'] = args.search_size
    data = [args.search]
    classif = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    print classif
    near_names = []
    for nn in classif['body']['predictions'][0]['nns']:
        near_names.append(nn['uri'])
    print near_names
    print len(near_names)
    cv2.imshow('query',image_resize(args.search,224.0))
    cv2.waitKey(0)
    for n in near_names:
        cv2.imshow('res',image_resize(n,224.0))
        cv2.waitKey(0)
    
dd.delete_service(sname,clear='')
