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
parser.add_argument("--model_repo",help="path to model",default=os.getcwd() + '/model')

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
layer_size = 1000 # default output code size
width = height = 224
binarized = False
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)
ntrees = 100
metric = 'angular'  # or 'euclidean'

# creating ML service
model_repo = args.model_repo

model = {'repository':model_repo,'templates':'../templates/caffe/'}
parameters_input = {'connector':'image','width':width,'height':height}

parameters_mllib = {'nclasses':nclasses}

template_name = 'googlenet'
if not os.path.isfile(model_repo + '/' + template_name + '.prototxt'):
    parameters_mllib['template'] = 'googlenet'

parameters_output = {}
dd.put_service(sname,model,description,mllib,
               parameters_input,parameters_mllib,parameters_output,mltype)

# reset call params
parameters_input = {}
parameters_mllib = {'gpu':True,'extract_layer':extract_layer}
parameters_output = {'binarized':binarized}

if args.index:
    try:
        os.remove('names.bin')
    except:
        pass
    s = shelve.open('names.bin')
        
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
            if c == 0:
                layer_size = len(p['vals'])
                s['layer_size'] = layer_size
                t = AnnoyIndex(layer_size,metric) # prepare index
            t.add_item(c,p['vals'])
            s[str(c)] = p['uri']
            c = c + 1
        #if c >= 10000:
        #    break
    print 'building index...\n'
    print 'layer_size=',layer_size
    t.build(ntrees)
    t.save('index.ann')
    s.close()

if args.search:
    s = shelve.open('names.bin')
    u = AnnoyIndex(s['layer_size'],metric)
    u.load('index.ann')
    data = [args.search]
    classif = dd.post_predict(sname,data,parameters_input,parameters_mllib,parameters_output)
    near = u.get_nns_by_vector(classif['body']['predictions'][0]['vals'],args.search_size,include_distances=True)
    print near
    near_names = []
    for n in near[0]:
        near_names.append(s[str(n)])
    print near_names
    cv2.imshow('query',image_resize(args.search,224.0))
    cv2.waitKey(0)
    for n in near_names:
        cv2.imshow('res',image_resize(n,224.0))
        cv2.waitKey(0)
    
dd.delete_service(sname,clear='')
