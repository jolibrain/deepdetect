pipeline {
  agent any
  stages {
    stage('Build Caffe GPU') {
      parallel {
        stage('Build Caffe GPU') {
          steps {
            sh '''script
mkdir build
cd build
cmake .. 
make'''
          }
        }
        stage('Build Caffe2') {
          steps {
            sh '''script
mkdir build_caffe2
cd build_caffe2
cmake .. -DUSE_CAFFE2'''
          }
        }
      }
    }
    stage('cleanup') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
}