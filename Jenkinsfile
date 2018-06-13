pipeline {
  agent any
  stages {
    stage('Build Caffe GPU') {
      steps {
        sh '''script
mkdir build
cd build
cmake .. -DBUILD_TESTS=ON
make'''
      }
    }
    stage('Build Xgboost') {
      steps {
        sh '''script
cd build
cmake .. -DUSE_XGBOOST=ON -DBUILD_TESTS=ON
make'''
      }
    }
    stage('Build Caffe2') {
      steps {
        sh '''script
mkdir build_caffe2
cd build_caffe2
cmake .. -DBUILD_TESTS=ON -DUSE_CAFFE2=ON
make'''
      }
    }
    stage('Build simsearch') {
      steps {
        sh '''script
cd build
cmake .. -DUSE_SIMSEARCH=ON -DBUILD_TESTS=ON
make'''
      }
    }
    stage('Build tsne') {
      steps {
        sh '''script
cd build
cmake .. -DUSE_TSNE=ON -DBUILD_TESTS=ON
make'''
      }
    }
    stage('') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
}