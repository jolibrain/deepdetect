pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        sh '''
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON -DUSE_CUDNN=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_XGBOOST=ON -DUSE_CAFFE2=ON -DUSE_NCNN=ON -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
make -j24'''
      }
    }
    stage('Tests GPU') {
      steps {
        sh '''cd build
ctest -V -E "http" '''
      }
    }
    stage('cleanup') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
    stage('Notify Chat') {
      steps {
        rocketSend(avatar: 'beniz', channel: 'dev', message: 'Build Completed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)')
      }
    }
  }
}