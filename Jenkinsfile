pipeline {
  agent any
  stages {
    stage('Build') {
      steps {
        sh '''script
mkdir build
cd build
cmake .. 
make'''
      }
    }
    stage('cleanup') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
}