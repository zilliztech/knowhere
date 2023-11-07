int total_timeout_minutes = 60
def knowhere_wheel=''
pipeline {
    agent {
        kubernetes {
            inheritFrom 'default'
            yamlFile 'ci/pod/e2e-cpu.yaml'
            defaultContainer 'main'
        }
    }

    options {
        timeout(time: total_timeout_minutes, unit: 'MINUTES')
        buildDiscarder logRotator(artifactDaysToKeepStr: '30')
        parallelsAlwaysFailFast()
        disableConcurrentBuilds(abortPrevious: true)
        preserveStashes(buildCount: 10)
    }
    stages {
        stage("Build"){
            steps {
                container("build"){
                    script{
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: "echo ${env.GIT_COMMIT} | cut -b 1-7 ").trim()
                        version="${env.CHANGE_ID}.${date}.${gitShortCommit}"
                        sh "apt-get update || true"
                        sh "apt-get install libaio-dev libopenblas-dev libcurl4-openssl-dev libdouble-conversion-dev libevent-dev libgflags-dev git -y"
                        sh "pip3 install conan==1.61.0"
                        sh "conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local"
                        sh "rm -rf /usr/local/lib/cmake/"
                        sh "mkdir build"
                        sh "cd build/ && conan install .. --build=missing -o with_diskann=True -s compiler.libcxx=libstdc++11 && conan build .."
                        sh "cd python && VERSION=${version} python3 setup.py bdist_wheel"
                        dir('python/dist'){
                        knowhere_wheel=sh(returnStdout: true, script: 'ls | grep .whl').trim()
                        archiveArtifacts artifacts: "${knowhere_wheel}", followSymlinks: false
                        }
                        // stash knowhere info for rebuild E2E Test only
                        sh "echo ${knowhere_wheel} > knowhere.txt"
                        stash includes: 'knowhere.txt', name: 'knowhereWheel'
                    }
                }
            }
        }
        stage("Test"){
            steps {
                script{
                    if ("${knowhere_wheel}"==''){
                        dir ("knowhereWheel"){
                            try{
                                unstash 'knowhereWheel'
                                knowhere_wheel=sh(returnStdout: true, script: 'cat knowhere.txt | tr -d \'\n\r\'')
                            }catch(e){
                                error "No knowhereWheel info remained ,please rerun build to build new package."
                            }
                        }
                    }
                    checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [],
                    userRemoteConfigs: [[credentialsId: 'milvus-ci', url: 'https://github.com/milvus-io/knowhere-test.git']]])
                    dir('tests'){
                      unarchive mapping: ["${knowhere_wheel}": "${knowhere_wheel}"]
                      sh "apt-get update || true"
                      sh "apt-get install dirmngr -y"
                      sh "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA"
                      sh "apt install python3-pip -y"
                      sh "apt install libopenblas-dev libaio-dev libdouble-conversion-dev libevent-dev -y"
                      sh "nvidia-smi"
                      sh "pip3 install ${knowhere_wheel} \
                          && cat requirements.txt | xargs -n 1 pip3 install && pytest -v -m 'L0'"
                    }
                }
            }
            post{
                always {
                    script{
                        sh 'cp /tmp/knowhere_ci.log knowhere_ci.log'
                        archiveArtifacts artifacts: 'knowhere_ci.log', followSymlinks: false
                    }
                }
            }
        }
    }
}
