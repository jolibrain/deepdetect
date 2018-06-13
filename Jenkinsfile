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
      parallel {
        stage('Build Xgboost') {
          steps {
            sh '''script
mkdir build_xgboost
cd build_xgboost
cmake .. -DUSE_XGBOOST=ON -DBUILD_TESTS=ON
make'''
          }
        }
        stage('Test Caffe GPU') {
          steps {
            sh '''cd build
ctest'''
          }
        }
      }
    }
    stage('Build Caffe2') {
      parallel {
        stage('Build Caffe2') {
          steps {
            sh '''script
mkdir build_caffe2
cd build_caffe2
cmake .. -DBUILD_TESTS=ON -DUSE_CAFFE2=ON
make'''
          }
        }
        stage('Test Xgboost') {
          steps {
            sh '''script
cd build_xgboost/tests
./ut_xgbapi
'''
          }
        }
      }
    }
    stage('Test Caffe2') {
      steps {
        sh '''script
cd build_caffe2/tests
./ut_caffe2api'''
      }
    }
    stage('Cleanup') {
      steps {
        cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
      }
    }
  }
}