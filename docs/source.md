# DeepDetect installation from source

Below are instructions for 20.04 LTS. For other Linux and Unix systems, steps may differ, CUDA, Caffe and other libraries may prove difficult to setup.

Beware of [dependencies](https://github.com/jolibrain/deepdetect/tree/master/docs/dependencies.md), typically on Debian/Ubuntu Linux, do:
```
sudo apt-get install build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libboost-all-dev libboost-iostreams-dev libcurlpp-dev libcurl4-openssl-dev libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev cmake libgoogle-perftools-dev unzip python-setuptools python3-dev python3-six libarchive-dev rapidjson-dev libmapbox-variant-dev wget libboost-test-dev libboost-stacktrace-dev python3-typing-extensions python3-numpy
```

## Default GPU build with OpenCV CUDA support
For compiling along with OpenCV:

```
mkdir build & cd build
DEEPDETECT_RELEASE=OFF DEEPDETECT_ARCH=gpu DEEPDETECT_BUILD=torch DEEPDETECT_DEFAULT_MODELS=false BUILD_OPENCV=ON ../build.sh
```

If you are building for one or more GPUs, you may need to add CUDA to your ld path:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

## Default CPU build

```
DEEPDETECT_RELEASE=OFF DEEPDETECT_ARCH=cpu DEEPDETECT_BUILD=torch DEEPDETECT_DEFAULT_MODELS=false BUILD_OPENCV=OFF ../build.sh
```

## Build for Jetson Orin

```
DEEPDETECT_RELEASE=OFF DEEPDETECT_ARCH=jetson DEEPDETECT_BUILD=tensorrt DEEPDETECT_DEFAULT_MODELS=false BUILD_OPENCV=ON BUILD_TESTS=ON ../build.sh
```

## Choosing interfaces :

DeepDetect can be used:
- from command line using the JSON API. To build the executable use:
```
cmake .. -DUSE_COMMAND_LINE=ON -DUSE_JSON_API=ON
```
- as a REST server (using JSON API). To build the server executable use (`USE_JSON_API` is auto-selected):
```
cmake .. -DUSE_HTTP_SERVER_OATPP=ON
```
- linked into another executable. To build only the library (and use a `dd::DeepDetect<dd::JSonAPI>` object in your own code) use:
```
cmake .. -DUSE_JSON_API=ON -DUSE_HTTP_SERVER_OATPP=OFF -DUSE_COMMAND_LINE=OFF

```

## Build with XGBoost support

If you would like to build with XGBoost, include the `-DUSE_XGBOOST=ON` parameter to `cmake`:
```
cmake .. -DUSE_XGBOOST=ON
```

If you would like to build the GPU support for XGBoost (experimental from DMLC), use the `-DUSE_XGBOOST_GPU=ON` parameter to `cmake`:
```
cmake .. -DUSE_XGBOOST=ON -DUSE_XGBOOST_GPU=ON
```

## Build with T-SNE support

Simply specify the option via cmake command line:
```
cmake .. -DUSE_TSNE=ON
```

## Build with Dlib support
Specify the following option via cmake:
```$xslt
cmake .. -DUSE_DLIB=ON
```
This will automatically build with GPU support if possible. Note: this will also enable cuDNN if available by default.

If you would like to constrain Dlib to CPU, use:
```
cmake .. -DUSE_DLIB=ON -DUSE_DLIB_CPU_ONLY=ON
```

## Build with TensorRT support

Some NVidia libraires from TensorRT need to be installed first:
```
apt install libnvinfer-plugin-dev libnvparsers-dev libnvonnxparsers-dev
```

## Build with TensorRT support + TRT oss parts
Specify the following option via cmake:
```$xslt
cmake .. -DUSE_TENSORRT=ON -DUSE_TENSORRT_OSS=ON
```
This compiles against https://github.com/NVIDIA/TensorRT , ie opensource parts (mainly parsers)

## Build with Libtorch support

Specify the following option via cmake:
```
cmake .. -DUSE_TORCH=ON
```
If you call cmake with the `-DUSE_CPU_ONLY` option, a cpu-only version of libtorch will be used.

## Build with similarity search support

Specify the following option via cmake:
```
cmake .. -DUSE_SIMSEARCH=ON
```

## Build with logs output into syslog

Specify the following option via cmake:
```
cmake .. -DUSE_DD_SYSLOG=ON
```

## Run tests

Note: running tests requires the automated download of ~75Mb of datasets, and computations may take around thirty minutes on a CPU-only machines.

To prepare for tests, install numpy:
```
sudo apt install python-numpy libgtest-dev
```
then compile with:
```
cmake -DBUILD_TESTS=ON ..
make
```
Run tests with:
```
ctest
```

## Code Style Rules

`clang-format` is used to enforce code style.

You can automatically format it with:

```
cmake ..
make clang-format
```

Or use your favorite editor with a clang-format plugin.

## Start the server

```
cd build/main
./dede

DeepDetect [ commit 73d4e638498d51254862572fe577a21ab8de2ef1 ]
Running DeepDetect HTTP server on localhost:8080
```

Main options are:
- `-host` to select which host to run on, default is `localhost`, use `0.0.0.0` to listen on all interfaces
- `-port` to select which port to listen to, default is `8080`
- `-nthreads` to select the number of HTTP threads, default is `10`

To see all options, do:
```
./dede --help
```

## Services auto-start

A list of services can be stored into a file and passed to the `dede` server so that they are all created upon server start. A list fo predictions can also be run automatically upon server start. The file is passed with:

```
./dede -service_start_list <yourfile>
```

File format is as follows:

- service creation:
```
service_create;sname;JSON string
```
where `sname` is the service name and the JSON is a string without external quotes

- service prediction
```
service_predict;JSON string
```

## Pure command line JSON API

To use deepdetect without the client/server architecture while passing the exact same JSON messages from the API:

```
./dede --jsonapi 1 <other options>
```

where `<other options>` stands for the command line parameters from the command line JSON API:

```
-info (/info JSON call) type: bool default: false
-service_create (/service/service_name call JSON string) type: string default: ""
-service_delete (/service/service_name DELETE call JSON string) type: string default: ""
-service_name (service name string for JSON call /service/service_name) type: string default: ""
-service_predict (/predict POST call JSON string) type: string default: ""
-service_train (/train POST call JSON string) type: string default: ""
-service_train_delete (/train DELETE call JSON string) type: string default: ""
-service_train_status (/train GET call JSON string) type: string default: ""

```

The options above can be obtained from running

```
./dede --help
```

Example of creating a service then listing it:

```
./dede --jsonapi 1 --service_name test --service_create '{"mllib":"caffe","description":"classification service","type":"supervised","parameters":{"input":{"connector":"image"},"mllib":{"template":"googlenet","nclasses":10}},"model":{"templates":"/path/to/deepdetect/templates/caffe/","repository":"/path/to/model/"}}'
```

Note that in command line mode the `--service_xxx` calls are executed sequentially, and synchronously. Also note the logs are those from the server, the JSON API response is not available in pure command line mode.
