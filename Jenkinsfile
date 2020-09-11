
def context = "Jenkins - Unit tests GPU"

def setBuildStatus(String message, String state) {
    step([
            $class: "GitHubCommitStatusSetter",
            contextSource: [$class: "ManuallyEnteredCommitContextSource", context: context],
            errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
            statusResultSource: [$class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        setBuildStatus("Building", "PENDING")
        sh '''
export PATH="/usr/lib/ccache/:$PATH"
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON -DUSE_CUDNN=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_XGBOOST=ON -DUSE_TORCH=ON -DUSE_NCNN=ON -DUSE_TENSORRT=ON -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
make -j24
ccache -s
'''
      }
    }
    stage('Tests GPU') {
      steps {
        setBuildStatus("Testing", "PENDING")
        sh '''cd build
ctest -V -E "http" '''
      }
    }
    stage('cleanup') {
      steps {
        setBuildStatus("Cleaning", "PENDING")
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
  post {
      success {
          setBuildStatus("Finished with no errors", "SUCCESS")
          rocketSend(channel: 'build', message: 'Build succeed')
      }
      unsuccessful {
          setBuildStatus("The job have failed", "FAILURE")
          rocketSend(channel: 'build', message: 'Build failed')
      }
  }
}
