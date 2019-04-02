"""
DeepDetect benchmark tool

Licence:
Copyright (c) 2017 Emmanuel Benazera

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

"""
This is a benchmark for image services with DeepDetect.

Instructions:
- wget https://deepdetect.com/stuff/bench.tar.gz
- untar bench.tar.gz onto the machine where your DeepDetect server runs
- remotely benchmark your server with (adapt options as needed):
  
  python dd_bench.py --host yourhost --port 8080 --sname faces --img-width 300 --img-height 300 --gpu --remote-bench-data-dir /server/path/to/bench/ --max-batch-size 64
"""

import time
import sys
import argparse
import csv
from dd_client import DD

parser = argparse.ArgumentParser(description='DeepDetect benchmark tool')
parser.add_argument('--host',help='server host',default='localhost')
parser.add_argument('--port',help='server port',type=int,default=8080)
parser.add_argument('--sname',help='service name')
parser.add_argument('--img-width',help='image width',type=int,default=224)
parser.add_argument('--img-height',help='image height',type=int,default=224)
parser.add_argument('--gpu',help='whether to bench GPU',action='store_true')
parser.add_argument('--gpuid',help='gpu id to use',type=int,default=0)
parser.add_argument('--cpu',help='whether to bench CPU',action='store_true')
parser.add_argument('--remote-bench-data-dir',help='when bench data directory, when available remotely on the server')
parser.add_argument('--max-batch-size',help='max batch size to be tested',type=int,default=256)
parser.add_argument('--list-bench-files',help='file holding the list of bench files',default='list_bench_files.txt')
parser.add_argument('--npasses',help='number of passes for every batch size',type=int,default=5)
parser.add_argument('--detection',help='whether benching a detection model',action='store_true')
parser.add_argument('--search',help='whether benching a similarity search service',action='store_true')
parser.add_argument('--search-multibox',help='whether benching a multibox similarity search service',action='store_true')
parser.add_argument('--create',help='model\'s folder name to create a service')
parser.add_argument('--nclasses',help='number of classes for service creation',type=int,default=1000)
parser.add_argument('--auto-kill',help='auto kill the service after benchmarking',action='store_true')
parser.add_argument('--csv-output',help='CSV file output')
args = parser.parse_args()

host = args.host
port = args.port
dd = DD(host,port)
dd.set_return_format(dd.RETURN_PYTHON)
autokill = args.auto_kill

# Create a service
if args.create:
  description = 'image classification service'
  mllib = 'caffe'
  model = {'repository':args.create}
  parameters_input = {'connector':'image','width':args.img_width,'height':args.img_height}
  parameters_mllib = {'nclasses':args.nclasses}
  parameters_output = {}
  dd.put_service(args.sname,model,description,mllib,
                 parameters_input,parameters_mllib,parameters_output)
else:
    pass

out_csv = None
csv_writer = None
if args.csv_output:
  out_csv = open(args.csv_output,'w+')
  csv_writer = csv.writer(out_csv)
  csv_writer.writerow(['batch_size','mean processing time','mean time per img'])
  
list_bench_files = []
with open(args.list_bench_files) as f:
    for l in f:
        list_bench_files.append(args.remote_bench_data_dir + '/' + l.rstrip())
init_batch_size = 1
batch_sizes = []
l = init_batch_size
while l <= args.max_batch_size:
    batch_sizes.append(l)
    if l < 32:
        l = l * 2
    else:
        l += 16

parameters_input = {}
parameters_mllib = {'gpu':args.gpu,'gpuid':args.gpuid}
parameters_output = {}
if args.detection:
    parameters_output['confidence_threshold'] = 0.1
    if args.search or args.search_multibox:
      parameters_output['search'] = True
      parameters_output['rois'] = 'rois'
      parameters_output['bbox'] = False
    else:
      parameters_output['bbox'] = True
    if args.search_multibox:
      parameters_output['multibox_rois'] = True
elif args.search:
  parameters_output['search'] = True
      
# First call to load model
data = list_bench_files[:1]
classif = dd.post_predict(args.sname,data,parameters_input,parameters_mllib,parameters_output)

for b in batch_sizes:
    data = list_bench_files[:b]
    #print data
    fail = False
    mean_ptime = 0
    mean_ptime_per_img = 0
    for i in range(0,args.npasses+1):
        #print 'testing batch size =',len(data)
        classif = dd.post_predict(args.sname,data,parameters_input,parameters_mllib,parameters_output)
        if classif['status']['code'] == 200:
            if i == 0:
                continue # skipping first pass so that the batch resize does not affect timing
            ptime = classif['head']['time']
            ptime_per_img = ptime/b
            mean_ptime += ptime
            mean_ptime_per_img += ptime_per_img
            print 'pass',i,' batch size =',b,' / processing time =',ptime, ' / time per image = ', ptime_per_img
        else:
            print classif['status']
            # reload model
            data =list_bench_files[:1]
            classif = dd.post_predict(args.sname,data,parameters_input,parameters_mllib,parameters_output)
            fail = True
            break
    mean_processing_time = mean_ptime/args.npasses
    mean_time_per_img = mean_ptime_per_img/args.npasses
    print '>>> batch size =',b,' / mean processing time =',mean_ptime/args.npasses, ' / mean time per image =',mean_ptime_per_img/args.npasses, ' / fail =',fail
    if args.csv_output:
      csv_writer.writerow([b,mean_processing_time,mean_time_per_img])
    #break

if autokill:
  dd.delete_service(args.sname)
  
