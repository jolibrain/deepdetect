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
  }
}