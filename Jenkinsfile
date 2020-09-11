pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        sh '''
export PATH="/usr/lib/ccache/:$PATH"
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON -DUSE_CUDNN=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_XGBOOST=ON -DUSE_TORCH=ON -DUSE_NCNN=ON -DUSE_TENSORRT=ON -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
make clang-format-check
make -j24
ccache -s
'''
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
  }
  post {
      success {
          rocketSend(channel: 'build', message: 'Build succeed')
      }
      unsuccessful {
          rocketSend(channel: 'build', message: 'Build failed')
      }
  }
}
