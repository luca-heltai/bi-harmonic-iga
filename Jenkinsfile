pipeline {
  agent {
    docker {
      image 'heltai/dealii:v9.1.1-gcc-mpi'
      args '--user $(id -u):$(id -g) -i -t --rm -P '
    }

  }
  stages {
    stage('Indent') {
      steps {
        sh './scripts/check_indentation.sh'
      }
    }
  }
}
