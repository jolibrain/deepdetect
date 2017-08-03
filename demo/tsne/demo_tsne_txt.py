from dd_client import DD
import matplotlib
import numpy as np
import time
import argparse
import sys

import matplotlib.pyplot as plt
import pylab

model_repo = "/tmp"
host = 'localhost'
port = 8080
sname = 'test'
description = 'clustering'
mllib = 'tsne'
dd = DD(host)
dd.set_return_format(dd.RETURN_PYTHON)

parser = argparse.ArgumentParser()
parser.add_argument("--txtdir",help="directory containing text files")
parser.add_argument("--perplexity",help="TSNE perplexity",type=int,default=30)
args = parser.parse_args()

training_repo = args.txtdir

# service creation
model = {'repository':model_repo}
parameters_input = {'connector':'txt'}
parameters_mllib = {}
parameters_output = {}
try:
    creat = dd.put_service(sname,model,description,mllib,
                           parameters_input,parameters_mllib,parameters_output,'unsupervised')
except:
    pass

# training
train_data = [training_repo]
parameters_input = {'min_count':10,'min_word_length':5}
parameters_mllib = {'iterations':500,'perplexity':args.perplexity}
parameters_output = {}
predout = dd.post_train(sname,train_data,parameters_input,parameters_mllib,parameters_output,async=True)

time.sleep(1)
train_status = ''
while True:
    train_status = dd.get_train(sname,job=1,timeout=3)
    if train_status['head']['status'] == 'running':
        print train_status['body']['measure']
    else:
        print train_status
        predout = train_status
        break

predictions = predout['body']['predictions']
N = len(predictions)
points = np.empty((N,2),dtype=np.float)
i = 0
for p in predictions:
    points[i,0] = p['vals'][0]
    points[i,1] = p['vals'][1]
    i = i + 1
        
pylab.xlim([-30,30])
pylab.ylim([-30,30])

plt.ioff()
colors = np.random.rand(N)
plt.scatter(points[:,0],points[:,1],colors)
pylab.show()

dd.delete_service(sname,clear='')
