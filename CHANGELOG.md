# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.24.0](https://github.com/jolibrain/deepdetect/compare/v0.23.1...v0.24.0) (2023-03-31)


### Features

* add custom api path to swagger ([4fe0df7](https://github.com/jolibrain/deepdetect/commit/4fe0df721d1cc7a1cd03870e473e744c4924bb58))
* add percent error measure display ([1cc15d6](https://github.com/jolibrain/deepdetect/commit/1cc15d6a50a00218a47f43387ca88b94ac665801))
* **api:** add a model_stats field containing the number of parameters of the model ([b562fee](https://github.com/jolibrain/deepdetect/commit/b562fee5834402a720ca54a25ef7de4c6026f036))
* **api:** add labels in service info ([66cbff5](https://github.com/jolibrain/deepdetect/commit/66cbff59ae8e071ba84f0c84edda7375c7a0d0cb))
* **api:** increase accepted header size ([07f6ff3](https://github.com/jolibrain/deepdetect/commit/07f6ff32a834ae20e79a4e9933d66e89392b2385))
* log model parameters and size at service startup ([041b649](https://github.com/jolibrain/deepdetect/commit/041b6493aec803a2e4e76fe91e80b17b94f94c4e))
* **regression:** add l1 metric for regression ([c82f08d](https://github.com/jolibrain/deepdetect/commit/c82f08d82763ef20719ca7b36d02f67ec69d0d78))
* **torch:** add radam optimizer ([5bba045](https://github.com/jolibrain/deepdetect/commit/5bba045ccd75ff13f85ad88160558e38c4410cba))
* **torch:** add translation and bbox duplication to data augmentation ([8752e1f](https://github.com/jolibrain/deepdetect/commit/8752e1f2f723a43194049cf570b42deca8ed8b5d))
* **torch:** allow data aug to be only noise or distort ([5a02234](https://github.com/jolibrain/deepdetect/commit/5a02234ce4a1f571759d3af1f31f818c36809798))
* **torch:** allow data augmentation w/o db ([f5b16b3](https://github.com/jolibrain/deepdetect/commit/f5b16b3f111dbd8c1555a15e6afe78dac61354b2))
* **torch:** data augmentation w/o db for bbox ([a99ca7b](https://github.com/jolibrain/deepdetect/commit/a99ca7b14a0d7e0eff5c816a8c959fff31b12ff1))
* **torch:** set data augmentation factors as requested ([e26a775](https://github.com/jolibrain/deepdetect/commit/e26a7751c5bd1f5cdd8db6bc727776e38f05e8da))
* **torch:** update torch to 1.13 ([9c5da36](https://github.com/jolibrain/deepdetect/commit/9c5da3605c8cb751dd92f8d659887bdc19214877))
* **trt:** add int8 inference ([a212a8e](https://github.com/jolibrain/deepdetect/commit/a212a8e0088bb7965df1d4af70830ec082b8e8a9))
* **trt:** recompile engine if wrong version is detected ([0f0bb62](https://github.com/jolibrain/deepdetect/commit/0f0bb624afdf1b4bab6538e487bb02cce3b46801))
* upgrade to TensorRT 8.4.3 ([1132760](https://github.com/jolibrain/deepdetect/commit/113276006ae0a5ea28c0274f635ab2cbea3e2d9c))


### Bug Fixes

* **api:** re-add parameters in info call ([df318cb](https://github.com/jolibrain/deepdetect/commit/df318cb77c9760292b56ce8e38c3c8498f54152b))
* raise exception when a bbox file contains invalid classes ([3a82a9d](https://github.com/jolibrain/deepdetect/commit/3a82a9d8263a333998bc58ebf83a02cb933752e8))
* **readme:** correct docker tags for ci-master ([49dde89](https://github.com/jolibrain/deepdetect/commit/49dde89be309981532a2fb22e431d0bb005f8ab6))
* **regression:** fix eucl metric in case of thresholded metric ([a006615](https://github.com/jolibrain/deepdetect/commit/a006615abadd2f89d2f77464be1d39afc1cbf739))
* take into account false negatives when computing average precision ([11905eb](https://github.com/jolibrain/deepdetect/commit/11905eb8d51f8d34a6d1b32e32eef806737f27d3))
* **tensorrt:** clarify conditions to rebuild engine ([9d08b0a](https://github.com/jolibrain/deepdetect/commit/9d08b0a0644d3c14c62ecc5d5b5d1eb41792dbf2))
* **torch:** add measures to output event when training not done ([5714767](https://github.com/jolibrain/deepdetect/commit/5714767bcbe2bbefaa5c3ffce52c5fa322489fde))
* **torch:** avoid race condition when building alphabet ([b1accb7](https://github.com/jolibrain/deepdetect/commit/b1accb7362e71afd75f967186588806e1744341f))
* **torch:** correctly normalize l1 and l2 metrics in case of multi dim regression ([cc9a636](https://github.com/jolibrain/deepdetect/commit/cc9a636da86016e4e434377ee3de243991c82511))
* **torch:** data augmentation handle dummy bboxes correctly ([53d0c39](https://github.com/jolibrain/deepdetect/commit/53d0c394a9aedfa0452b6183721cc350babcb540))
* **torch:** dataset size is half the database size ([9541de1](https://github.com/jolibrain/deepdetect/commit/9541de16daf401ff298e9dece6673773e0637e4d))
* **torch:** make multi dim regression for images work ([00985bf](https://github.com/jolibrain/deepdetect/commit/00985bfb4ab36905dc021ff91336915483c21a8c))
* **torch:** small glitches in data augmentation ([678944a](https://github.com/jolibrain/deepdetect/commit/678944aa2a0f76c0efc326abe83da425aec04e21))
* **torch:** when reading bbox dataset, also check that the class is not >= nclasses ([7b2de88](https://github.com/jolibrain/deepdetect/commit/7b2de88495387dbaad9629757e208365be3f9e9a))
* **trace_yolox:** bbox shifted by 1 when training yolox ([487bad7](https://github.com/jolibrain/deepdetect/commit/487bad77c1e5ac1f9f6f295316dfb579c5b63ce1))
* **trace_yolox:** input shape for nonsquare images ([6db03be](https://github.com/jolibrain/deepdetect/commit/6db03bec32143b91ca22ac0a4fc74365780ae249))
* **trace_yolox:** make model autodetection work ([7340a48](https://github.com/jolibrain/deepdetect/commit/7340a486338cfe415b5d9187692da47c6bc0d4f2))
