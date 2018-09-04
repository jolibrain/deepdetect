### Caffe2 Tools

---

#### Compiled with DeepDetect

Compile them using both ```-DUSE_CAFFE2=ON``` and ```-DBUILD_TOOLS=ON```.

The binaries are placed in the "tools" subdirectory of the build.

**convert_proto** can be used to convert ".pb" files into ".pbtxt" or vice versa

```
./convert_proto -i net.pb -o human_readable_net.pbtxt
```

(```-i``` and ```-o``` stands for "input" and "output")

Use ```-t``` to specify the object type (set to "NetDef" by default)

```
./convert_proto -i mean.pb -o mean.pbtxt -t TensorProto
```

Finally, you can use ```-b``` (binary) to convert from human-readable to binary mode

```
./convert_proto -i mean.pbtxt -o mean.pb -t TensorProto -b
```

*(Tip : A human readable mean file looks like that)*
```
data_type: FLOAT
dims: 3
float_data: 102.9801
float_data: 115.9465
float_data: 122.7717
```

**net2template** can be used to reset a trained model into a deepdetect template.

(e.g. https://github.com/caffe2/models)

It takes 2 arguments : the input directory and the output directory

```
./net2template ./caffe2/models/resnet50 ./my_templates/r50
```

**net2svg** can be used to transform a net protobuf into an SVG graph.

It takes 2 arguments : the net's path and the SVG's path

```
./net2svg ./predict_net.pbtxt ./predict_net.svg
```

*/!\ Important /!\\*

*To prevent crashes and too heavy computations, a limit is set to 600 operators*
*(a resnet50 duplicated on 3 GPUS with all the gradients ... won't work).*

*Operators in an initialization net aren't linked with each others.*
*Thus the graph will appear as a long list of boxes aligned in one dimension, some viewer may not be able display it.*

---

#### Import / Export DeepDectect's models with caffe2 python

Every protobuf object is accessible with a single import

```
from caffe2.proto import caffe2_pb2

net = caffe2_pb2.NetDef()
op = caffe2_pb2.OperatorDef()
tensor = caffe2_pb2.TensorProto()
...
```

And can simply be loaded or exported from/to a ".pb" file:
```
def import_proto(proto, path):
    proto.ParseFromString(open(path, "rb").read())

def export_proto(proto, path):
    open(path, "wb").write(proto.SerializeToString())
```

The protobufs "predict_net" and "init_net" used in deepdetect correspond to the following
```
from caffe2.python import model_helper

m = model_helper.ModelHelper(name="my_net")

predict_net = m.net.Proto() # or m.net._net
init_net = m.param_init_net.Proto() # or m.param_init_net._net
```

#### Caffe2 tutorials

https://github.com/caffe2/tutorials

**To create a model**
```
from caffe2.python import model_helper, brew
```
See https://github.com/caffe2/tutorials/blob/master/MNIST.ipynb

**To execute a model**
```
from caffe2.python import workspace, core
```
See https://github.com/caffe2/tutorials/blob/master/Loading_Pretrained_Models.ipynb

**To train a model**
```
from caffe2.python import data_parallel_model, optimizer
```
See https://github.com/caffe2/tutorials/blob/master/Multi-GPU_Training.ipynb
