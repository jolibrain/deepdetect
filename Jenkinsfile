pipeline {
  agent any
  stages {
    stage('Build Caffe GPU') {
      steps {
        sh '''script
mkdir -p build
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
cd build
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
    stage('Tests') {
      steps {
        sh '''cd build
ctest'''
      }
    }
    stage('cleanup') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
}