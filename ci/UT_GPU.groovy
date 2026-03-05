int total_timeout_minutes = 240
def knowhere_wheel=''
pipeline {
    agent {
        kubernetes {
            cloud "new_ci_idc"
            inheritFrom 'default'
            yamlFile 'ci/pod/ut-gpu.yaml'
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
        stage("UT"){
            steps {
                container("build"){
                    script{
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: "echo ${env.GIT_COMMIT} | cut -b 1-7 ").trim()
                        version="${env.CHANGE_ID}.${date}.${gitShortCommit}"
                        sh "source scripts/ci_deps.sh && install_base_deps && setup_conan_remote"
                        // GPU-only: git not pre-installed in GPU build image
                        sh "apt-get install -y git"
                        sh "cmake --version"
                        sh "nvidia-smi --query-gpu=name --format=csv,noheader"
                        sh "make ut-gpu"
                    }
                }
            }
        }
    }
}
