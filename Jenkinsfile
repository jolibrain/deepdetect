pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        sh '''script
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON -DUSE_CUDNN=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_XGBOOST=ON -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
make'''
      }
    }
    stage('Tests GPU') {
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