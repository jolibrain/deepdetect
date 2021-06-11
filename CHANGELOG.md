# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.18.0](https://github.com/jolibrain/deepdetect/compare/v0.17.0...v0.18.0) (2021-06-11)


### Features

* **build:** CMake config file to link with dede ([dd71a35](https://github.com/jolibrain/deepdetect/commit/dd71a35df831bab5382e4ee5885b425d5364a3b9))
* add multigpu support for external native models ([90dcadd](https://github.com/jolibrain/deepdetect/commit/90dcaddc064a17275ddb709a4fe26ee690c7fc58))
* **ml:** inference for GAN generators with TensorRT backend ([c93188c](https://github.com/jolibrain/deepdetect/commit/c93188c7a89d7efbea0269345e32c90df29ef74a))
* **ml:** python script to trace timm vision models ([055fdfe](https://github.com/jolibrain/deepdetect/commit/055fdfe49d08a99b6b9379d3e2863dfff9ff8c1c))
* **predict:** add best_bbox for torch, trt, caffe, ncnn backend ([7890401](https://github.com/jolibrain/deepdetect/commit/7890401e1751d3ca855a48a1a5badd48fcac833f))
* **torch:** add dataloader_threads in API ([74a036d](https://github.com/jolibrain/deepdetect/commit/74a036d58b98059f4592102b7e54d90490773258))
* **torch:** add multigpu for torch models ([447dd53](https://github.com/jolibrain/deepdetect/commit/447dd532c8e8d996675a091c7f3875fecd793aed))
* **torch:** support detection models in chains ([7bb9705](https://github.com/jolibrain/deepdetect/commit/7bb9705fa4eeac3af34e0dd8bc94eab0224fc120))
* **TRT:** port to TensorRT 21.04/7.2.3 ([4377451](https://github.com/jolibrain/deepdetect/commit/4377451dcbad488d3ee30a6083a3f82fdee2b196))


### Bug Fixes

* moving back to FAISS master ([916338b](https://github.com/jolibrain/deepdetect/commit/916338b9611d7285dea6dec92cfdd6d3699d37dc))
* **build:** add required definitions and include directory for building external dd api ([a059428](https://github.com/jolibrain/deepdetect/commit/a059428357b01836f9efa0c83be0e79549d9774c))
* **build:** do not patch/rebuild tensorrt if not needed ([bfd29ec](https://github.com/jolibrain/deepdetect/commit/bfd29ec071207cb9d528c462046889f8a6cdcd3c))
* **build:** torch 1.8 with cuda 11.3 string_view patch ([5002308](https://github.com/jolibrain/deepdetect/commit/50023087bda036118b18c2fd8733a991be3ab39b))
* **chain:** fixed_size crops now work at the edges of images ([8e38e35](https://github.com/jolibrain/deepdetect/commit/8e38e35fc242db0459664ba13e90b0c16f18b5b5))
* **dto:** allow scale input param to be either bool for csv/csvts or float for img ([168fc7c](https://github.com/jolibrain/deepdetect/commit/168fc7cb0c1b018c7408cba01184543e89b64c58))
* **log:** typo in ncnn model log ([0163b02](https://github.com/jolibrain/deepdetect/commit/0163b02de3639fce9e9746335f7a96188b99ffa2))
* **ncnn:** fix ncnnapi deserialization error ([089aacd](https://github.com/jolibrain/deepdetect/commit/089aacde7693435b2c90952d4961b8d41d9668ea))
* **ncnn:** fix typo in ut ([893217b](https://github.com/jolibrain/deepdetect/commit/893217b1de938d2465ee72d14dd168ffed1a7800))
* **trt:** apply std to input image correctly ([23bd913](https://github.com/jolibrain/deepdetect/commit/23bd913ac180b56eddbf90c71d1f2e8bc2310c54))
