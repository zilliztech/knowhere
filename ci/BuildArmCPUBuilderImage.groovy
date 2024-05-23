// Uses Declarative syntax to run commands inside a container.
pipeline {
      agent {
            kubernetes {
                cloud 'kubernetes'
                defaultContainer 'main'
                yamlFile 'ci/pod/rte-arm.yaml'
                customWorkspace '/home/jenkins/agent/workspace'
            }
    }
    environment {
        CI_DOCKER_CREDENTIAL_ID = "dockerhub"
    }

    parameters{
        string(
            description: 'os(ubuntu22.04,ubuntu20.04,centos7,ubuntu18.04)',
            name: 'os',
            defaultValue: 'ubuntu22.04'
        )
    }
     stages {
        stage ('Build'){
            steps {
                container('main') {
                    script {
                        sh 'git config --global --add safe.directory /home/jenkins/agent/workspace'
                        sh 'ls -lah'
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                        def image="milvusdb/knowhere-cpu-build:arm64-${os}-${date}-${gitShortCommit}"
                        sh "docker build -t ${image} -f ci/docker/builder/cpu/${params.os}/arm64/Dockerfile ."
                        withCredentials([usernamePassword(credentialsId: "${env.CI_DOCKER_CREDENTIAL_ID}", usernameVariable: 'CI_REGISTRY_USERNAME', passwordVariable: 'CI_REGISTRY_PASSWORD')]){
                            sh "docker login -u ${CI_REGISTRY_USERNAME} -p ${CI_REGISTRY_PASSWORD}"
                            sh "docker push ${image}"
                        }
                    }
                }
            }
        }
    }
}
