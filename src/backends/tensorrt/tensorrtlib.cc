
// tensorrtlib.cc ---

// Copyright (C) 2019 Jolibrain http://www.jolibrain.com

// Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "outputconnectorstrategy.h"
#include "tensorrtlib.h"
#include "utils/apitools.h"
#include "tensorrtinputconns.h"
#include "utils/apitools.h"
#include "NvInferPlugin.h"
#include "protoUtils.h"
#include <cuda_runtime_api.h>
#include <string>

namespace dd
{

  static TRTLogger trtLogger;

  static int findEngineBS(std::string repo, std::string engineFileName)
  {
    std::unordered_set<std::string> lfiles;
    fileops::list_directory(repo, true, false, false, lfiles);
    for (std::string s : lfiles)
      {
        if (s.find(engineFileName) != std::string::npos)
          {
            std::string bs_str;
            for (auto it = s.crbegin(); it != s.crend(); ++it)
              {
                if (isdigit(*it))
                  bs_str += (*it);
                else
                  break;
              }
            std::reverse(bs_str.begin(), bs_str.end());
            if (bs_str.length() == 0)
              return -1;
            return std::stoi(bs_str);
          }
      }
    return -1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::TensorRTLib(const TensorRTModel &cmodel)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TensorRTModel>(cmodel)
  {
    this->_libname = "tensorrt";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::TensorRTLib(TensorRTLib &&tl) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TensorRTModel>(std::move(tl))
  {
    this->_libname = "tensorrt";
    _nclasses = tl._nclasses;
    _dla = tl._dla;
    _datatype = tl._datatype;
    _max_batch_size = tl._max_batch_size;
    _max_workspace_size = tl._max_workspace_size;
    _top_k = tl._top_k;
    _builder = tl._builder;
    _engineFileName = tl._engineFileName;
    _readEngine = tl._readEngine;
    _writeEngine = tl._writeEngine;
    _TRTContextReady = tl._TRTContextReady;
    _timeserie = tl._timeserie;
    _buffers = tl._buffers;
    _bbox = tl._bbox;
    _ctc = tl._ctc;
    _inputIndex = tl._inputIndex;
    _outputIndex0 = tl._outputIndex0;
    _outputIndex1 = tl._outputIndex1;
    _floatOut = tl._floatOut;
    _keepCount = tl._keepCount;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::~TensorRTLib()
  {
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::init_mllib(const APIData &ad)
  {
    trtLogger.setLogger(this->_logger);
    initLibNvInferPlugins(&trtLogger, "");

    if (ad.has("tensorRTEngineFile"))
      _engineFileName = ad.get("tensorRTEngineFile").get<std::string>();
    if (ad.has("readEngine"))
      _readEngine = ad.get("readEngine").get<bool>();
    if (ad.has("writeEngine"))
      _writeEngine = ad.get("writeEngine").get<bool>();

    if (ad.has("maxWorkspaceSize"))
      {
        _max_workspace_size
            = ad.get("maxWorkspaceSize").get<int>() * (1 << 20);
        this->_logger->info("setting max workspace size to {}",
                            _max_workspace_size);
      }
    if (ad.has("maxBatchSize"))
      {
        int nmbs = ad.get("maxBatchSize").get<int>();
        _max_batch_size = nmbs;
        this->_logger->info("setting max batch size to {}", _max_batch_size);
      }

    if (ad.has("dla"))
      _dla = ad.get("dla").get<int>();

    if (ad.has("datatype"))
      {
        std::string datatype = ad.get("datatype").get<std::string>();
        if (datatype == "fp32")
          _datatype = nvinfer1::DataType::kFLOAT;
        else if (datatype == "fp16")
          _datatype = nvinfer1::DataType::kHALF;
        else if (datatype == "int32")
          _datatype = nvinfer1::DataType::kINT32;
        else if (datatype == "int8")
          _datatype = nvinfer1::DataType::kINT8;
      }

    if (ad.has("gpuid"))
      _gpuid = ad.get("gpuid").get<int>();
    cudaSetDevice(_gpuid);

    model_type(this->_mlmodel._def, this->_mltype);

    _builder = std::shared_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(trtLogger),
        [=](nvinfer1::IBuilder *b) { b->destroy(); });
    if (_dla != -1)
      {
        if (_builder->getNbDLACores() == 0)
          this->_logger->info("Trying to use DLA core on a platform  that "
                              "doesn't have any DLA cores");
        else
          {
            if (_datatype == nvinfer1::DataType::kINT32)
              {
                this->_logger->info("asked for int32 on dla : forcing int8");
                _datatype = nvinfer1::DataType::kINT8;
              }
            else if (_datatype == nvinfer1::DataType::kFLOAT)
              {
                this->_logger->info(
                    "asked for float32 on dla : forcing float16");
                _datatype = nvinfer1::DataType::kHALF;
              }
            _builder->allowGPUFallback(true);
            if (_datatype == nvinfer1::DataType::kINT8)
              {
                _builder->setInt8Mode(true);
                _builder->setFp16Mode(false);
              }
            else
              {
                _builder->setInt8Mode(false);
                _builder->setFp16Mode(true);
              }
            _builder->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            _builder->setDLACore(_dla);
          }
      }
    else
      {
        if (_datatype == nvinfer1::DataType::kHALF)
          {
            _builder->setInt8Mode(false);
            if (_builder->platformHasFastFp16())
              {
                _builder->setFp16Mode(true);
                this->_logger->info("Setting FP16 mode");
              }
            else
              this->_logger->info("Platform does not has Fast TP16 mode");
          }
        else
          {
            _builder->setInt8Mode(false);
            _builder->setFp16Mode(false);
          }
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    nvcaffeparser1::shutdownProtobufLibrary();
    cudaFree(_buffers.data()[_inputIndex]);
    cudaFree(_buffers.data()[_outputIndex0]);
    if (_bbox)
      cudaFree(_buffers.data()[_outputIndex1]);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                  TMLModel>::train(const APIData &ad, APIData &out)
  {
    this->_logger->warn("Training not supported on tensorRT backend");
    (void)ad;
    (void)out;
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                  TMLModel>::predict(const APIData &ad, APIData &out)
  {
    std::lock_guard<std::mutex> lock(
        _net_mutex); // no concurrent calls since the net is not
                     // re-instantiated

    if (ad.getobj("parameters").getobj("mllib").has("gpuid"))
      _gpuid = ad.getobj("parameters").getobj("mllib").get("gpuid").get<int>();
    cudaSetDevice(_gpuid);

    APIData ad_output = ad.getobj("parameters").getobj("output");
    std::string out_blob = "prob";
    TInputConnectorStrategy inputc(this->_inputc);

    if (!_TRTContextReady)
      {
        if (ad_output.has("bbox"))
          _bbox = ad_output.get("bbox").get<bool>();
        // Ctc model
        if (ad_output.has("ctc"))
          {
            _ctc = ad_output.get("ctc").get<bool>();
            if (_ctc)
              {
                if (ad_output.has("blank_label"))
                  blank_label = ad_output.get("blank_label").get<int>();
              }
          }

        if (_bbox == true)
          out_blob = "detection_out";
        else if (_ctc == true)
          {
            out_blob = "probs";
            throw MLLibBadParamException(
                "ocr not yet implemented over tensorRT backend");
          }
        else if (_timeserie)
          {
            out_blob = "rnn_pred";
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }

        _nclasses = findNClasses(this->_mlmodel._def, _bbox);
        if (_bbox)
          _top_k = findTopK(this->_mlmodel._def);

        if (_nclasses <= 0)
          this->_logger->error("could not determine number of classes");

        bool engineRead = false;

        if (_readEngine)
          {
            int bs = findEngineBS(this->_mlmodel._repo, _engineFileName);
            if (bs != _max_batch_size && bs != -1)
              this->_logger->warn(
                  "found existing engine with max_batch_size {}, using it",
                  bs);
            std::ifstream file(this->_mlmodel._repo + "/" + _engineFileName
                                   + "_bs" + std::to_string(bs),
                               std::ios::binary);
            if (file.good())
              {
                std::vector<char> trtModelStream;
                size_t size{ 0 };
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream.resize(size);
                file.read(trtModelStream.data(), size);
                file.close();
                nvinfer1::IRuntime *runtime
                    = nvinfer1::createInferRuntime(trtLogger);
                _engine = std::shared_ptr<nvinfer1::ICudaEngine>(
                    runtime->deserializeCudaEngine(
                        trtModelStream.data(), trtModelStream.size(), nullptr),
                    [=](nvinfer1::ICudaEngine *e) { e->destroy(); });
                runtime->destroy();
                engineRead = true;
              }
          }

        if (!engineRead)
          {

            int fixcode
                = fixProto(this->_mlmodel._repo + "/" + "net_tensorRT.proto",
                           this->_mlmodel._def);
            switch (fixcode)
              {
              case 1:
                this->_logger->error(
                    "TRT backend could not open model prototxt");
                break;
              case 2:
                this->_logger->error(
                    "TRT backend  could not write transformed model prototxt");
                break;
              default:
                break;
              }

            nvinfer1::INetworkDefinition *network = _builder->createNetwork();
            nvcaffeparser1::ICaffeParser *caffeParser
                = nvcaffeparser1::createCaffeParser();

            const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor
                = caffeParser->parse(std::string(this->_mlmodel._repo + "/"
                                                 + "net_tensorRT.proto")
                                         .c_str(),
                                     this->_mlmodel._weights.c_str(), *network,
                                     _datatype);
            if (!blobNameToTensor)
              throw MLLibInternalException("Error while parsing caffe model "
                                           "for conversion to TensorRT");

            network->markOutput(*blobNameToTensor->find(out_blob.c_str()));

            if (out_blob == "detection_out")
              network->markOutput(*blobNameToTensor->find("keep_count"));
            _builder->setMaxBatchSize(_max_batch_size);
            _builder->setMaxWorkspaceSize(_max_workspace_size);

            network->getLayer(0)->setPrecision(nvinfer1::DataType::kFLOAT);

            nvinfer1::ILayer *outl = NULL;
            int idx = network->getNbLayers() - 1;
            while (outl == NULL)
              {
                nvinfer1::ILayer *l = network->getLayer(idx);
                if (strcmp(l->getName(), out_blob.c_str()) == 0)
                  {
                    outl = l;
                    break;
                  }
                idx--;
              }
            // force output to be float32
            outl->setPrecision(nvinfer1::DataType::kFLOAT);

            nvinfer1::ICudaEngine *le = _builder->buildCudaEngine(*network);
            _engine = std::shared_ptr<nvinfer1::ICudaEngine>(
                le, [=](nvinfer1::ICudaEngine *e) { e->destroy(); });

            if (_writeEngine)
              {
                std::ofstream p(this->_mlmodel._repo + "/" + _engineFileName
                                    + "_bs" + std::to_string(_max_batch_size),
                                std::ios::binary);
                nvinfer1::IHostMemory *trtModelStream = _engine->serialize();
                p.write(reinterpret_cast<const char *>(trtModelStream->data()),
                        trtModelStream->size());
                trtModelStream->destroy();
              }

            network->destroy();
            caffeParser->destroy();
          }

        _context = std::shared_ptr<nvinfer1::IExecutionContext>(
            _engine->createExecutionContext(),
            [=](nvinfer1::IExecutionContext *e) { e->destroy(); });
        _TRTContextReady = true;

        _inputIndex = _engine->getBindingIndex("data");
        _outputIndex0 = _engine->getBindingIndex(out_blob.c_str());

        if (_bbox)
          {
            _outputIndex1 = _engine->getBindingIndex("keep_count");
            _buffers.resize(3);
            _floatOut.resize(_max_batch_size * _top_k * 7);
            _keepCount.resize(_max_batch_size);
            if (inputc._bw)
              cudaMalloc(&_buffers.data()[_inputIndex],
                         _max_batch_size * inputc._height * inputc._width
                             * sizeof(float));
            else
              cudaMalloc(&_buffers.data()[_inputIndex],
                         _max_batch_size * 3 * inputc._height * inputc._width
                             * sizeof(float));
            cudaMalloc(&_buffers.data()[_outputIndex0],
                       _max_batch_size * _top_k * 7 * sizeof(float));
            cudaMalloc(&_buffers.data()[_outputIndex1],
                       _max_batch_size * sizeof(int));
          }
        else if (_ctc)
          {
            throw MLLibBadParamException(
                "ocr not yet implemented over tensorRT backend");
          }
        else if (_timeserie)
          {
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else // classification
          {
            _buffers.resize(2);
            _floatOut.resize(_max_batch_size * this->_nclasses);
            if (inputc._bw)
              cudaMalloc(&_buffers.data()[_inputIndex],
                         _max_batch_size * inputc._height * inputc._width
                             * sizeof(float));
            else
              cudaMalloc(&_buffers.data()[_inputIndex],
                         _max_batch_size * 3 * inputc._height * inputc._width
                             * sizeof(float));
            cudaMalloc(&_buffers.data()[_outputIndex0],
                       _max_batch_size * _nclasses * sizeof(float));
          }
      }

    APIData cad = ad;

    TOutputConnectorStrategy tout(this->_outputc);
    try
      {
        inputc.transform(cad);
      }
    catch (...)
      {
        throw;
      }

    int idoffset = 0;
    std::vector<APIData> vrad;

    cudaSetDevice(_gpuid);
    cudaStream_t cstream;
    cudaStreamCreate(&cstream);

    while (true)
      {

        int num_processed = inputc.process_batch(_max_batch_size);
        if (num_processed == 0)
          break;

        try
          {
            if (_bbox)
              {
                if (inputc._bw)
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
                else
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * 3 * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
                _context->enqueue(num_processed, _buffers.data(), cstream,
                                  nullptr);
                cudaMemcpyAsync(_floatOut.data(),
                                _buffers.data()[_outputIndex0],
                                num_processed * _top_k * 7 * sizeof(float),
                                cudaMemcpyDeviceToHost, cstream);
                cudaMemcpyAsync(_keepCount.data(),
                                _buffers.data()[_outputIndex1],
                                num_processed * sizeof(int),
                                cudaMemcpyDeviceToHost, cstream);
                cudaStreamSynchronize(cstream);
              }
            else if (_ctc)
              {
                throw MLLibBadParamException(
                    "ocr not yet implemented over tensorRT backend");
              }
            else if (_timeserie)
              {
                throw MLLibBadParamException(
                    "timeseries not yet implemented over tensorRT backend");
              }
            else // classification
              {
                if (inputc._bw)
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
                else
                  cudaMemcpyAsync(_buffers.data()[_inputIndex], inputc.data(),
                                  num_processed * 3 * inputc._height
                                      * inputc._width * sizeof(float),
                                  cudaMemcpyHostToDevice, cstream);
                _context->enqueue(num_processed, _buffers.data(), cstream,
                                  nullptr);
                cudaMemcpyAsync(_floatOut.data(),
                                _buffers.data()[_outputIndex0],
                                num_processed * _nclasses * sizeof(float),
                                cudaMemcpyDeviceToHost, cstream);
                cudaStreamSynchronize(cstream);
              }
          }
        catch (std::exception &e)
          {
            this->_logger->error("Error while processing forward TRT pass, "
                                 "not enough memory ? {}",
                                 e.what());
            throw;
          }

        std::vector<double> probs;
        std::vector<std::string> cats;
        std::vector<APIData> bboxes;
        std::vector<APIData> series;

        // Get confidence_threshold
        float confidence_threshold = 0.0;
        if (ad_output.has("confidence_threshold"))
          {
            apitools::get_float(ad_output, "confidence_threshold",
                                confidence_threshold);
          }

        if (_bbox)
          {
            int results_height = _top_k;
            const int det_size = 7;

            const float *outr = _floatOut.data();

            for (int j = 0; j < num_processed; j++)
              {
                int k = 0;
                std::vector<double> probs;
                std::vector<std::string> cats;
                std::vector<APIData> bboxes;
                APIData rad;
                std::string uri = inputc._ids.at(idoffset + j);
                auto bit = inputc._imgs_size.find(uri);
                int rows = 1;
                int cols = 1;
                if (bit != inputc._imgs_size.end())
                  {
                    // original image size
                    rows = (*bit).second.first;
                    cols = (*bit).second.second;
                  }
                else
                  {
                    this->_logger->error(
                        "couldn't find original image size for {}", uri);
                  }
                bool leave = false;
                int curi = -1;
                while (true && k < results_height)
                  {
                    if (outr[0] == -1)
                      {
                        // skipping invalid detection
                        this->_logger->error("skipping invalid detection");
                        outr += det_size;
                        leave = true;
                        break;
                      }
                    std::vector<float> detection(outr, outr + det_size);
                    if (curi == -1)
                      curi = detection[0]; // first pass
                    else if (curi != detection[0])
                      break; // this belongs to next image
                    ++k;
                    outr += det_size;
                    if (detection[2] < confidence_threshold)
                      continue;

                    // Fix border of bboxes
                    detection[3] = std::max(((float)detection[3]), 0.0f);
                    detection[4] = std::max(((float)detection[4]), 0.0f);
                    detection[5] = std::min(((float)detection[5]), 1.0f);
                    detection[6] = std::min(((float)detection[6]), 1.0f);

                    probs.push_back(detection[2]);
                    cats.push_back(this->_mlmodel.get_hcorresp(detection[1]));
                    APIData ad_bbox;
                    ad_bbox.add("xmin", static_cast<double>(detection[3]
                                                            * (cols - 1)));
                    ad_bbox.add("ymin", static_cast<double>(detection[4]
                                                            * (rows - 1)));
                    ad_bbox.add("xmax", static_cast<double>(detection[5]
                                                            * (cols - 1)));
                    ad_bbox.add("ymax", static_cast<double>(detection[6]
                                                            * (rows - 1)));
                    bboxes.push_back(ad_bbox);
                  }
                if (leave)
                  continue;
                rad.add("uri", uri);
                rad.add("loss", 0.0); // XXX: unused
                rad.add("probs", probs);
                rad.add("cats", cats);
                rad.add("bboxes", bboxes);
                vrad.push_back(rad);
              }
          }

        else if (_ctc)
          {
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else if (_timeserie)
          {
            throw MLLibBadParamException(
                "timeseries not yet implemented over tensorRT backend");
          }
        else // classification
          {
            for (int j = 0; j < num_processed; j++)
              {
                APIData rad;
                if (!inputc._ids.empty())
                  rad.add("uri", inputc._ids.at(idoffset + j));
                else
                  rad.add("uri", std::to_string(idoffset + j));
                rad.add("loss", 0.0);
                std::vector<double> probs;
                std::vector<std::string> cats;

                for (int i = 0; i < _nclasses; i++)
                  {
                    double prob = _floatOut.at(j * _nclasses + i);
                    if (prob < confidence_threshold)
                      continue;
                    probs.push_back(prob);
                    cats.push_back(this->_mlmodel.get_hcorresp(i));
                  }

                rad.add("probs", probs);
                rad.add("cats", cats);
                vrad.push_back(rad);
              }
          }
        idoffset += num_processed;
      }

    cudaStreamDestroy(cstream);

    tout.add_results(vrad);

    out.add("nclasses", this->_nclasses);
    if (_bbox)
      out.add("bbox", true);
    out.add("roi", false);
    out.add("multibox_rois", false);
    tout.finalize(ad.getobj("parameters").getobj("output"), out,
                  static_cast<MLModel *>(&this->_mlmodel));

    if (ad.has("chain") && ad.get("chain").get<bool>())
      {
        if (typeid(inputc) == typeid(ImgTensorRTInputFileConn))
          {
            APIData chain_input;
            if (!reinterpret_cast<ImgTensorRTInputFileConn *>(&inputc)
                     ->_orig_images.empty())
              chain_input.add(
                  "imgs", reinterpret_cast<ImgTensorRTInputFileConn *>(&inputc)
                              ->_orig_images);
            else
              chain_input.add(
                  "imgs", reinterpret_cast<ImgTensorRTInputFileConn *>(&inputc)
                              ->_images);
            chain_input.add(
                "imgs_size",
                reinterpret_cast<ImgTensorRTInputFileConn *>(&inputc)
                    ->_images_size);
            out.add("input", chain_input);
          }
      }

    out.add("status", 0);
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TensorRTLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::model_type(const std::string &param_file,
                                         std::string &mltype)
  {
    std::ifstream paramf(param_file);
    std::stringstream content;
    content << paramf.rdbuf();

    std::size_t found_detection = content.str().find("DetectionOutput");
    if (found_detection != std::string::npos)
      {
        mltype = "detection";
        return;
      }
    std::size_t found_ocr = content.str().find("ContinuationIndicator");
    if (found_ocr != std::string::npos)
      {
        mltype = "ctc";
        return;
      }
    mltype = "classification";
  }

  template class TensorRTLib<ImgTensorRTInputFileConn, SupervisedOutput,
                             TensorRTModel>;

}
