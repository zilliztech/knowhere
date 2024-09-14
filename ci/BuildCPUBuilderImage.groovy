// Uses Declarative syntax to run commands inside a container.
pipeline {
    agent {
        kubernetes {
            // Rather than inline YAML, in a multibranch Pipeline you could use: yamlFile 'jenkins-pod.yaml'
            // Or, to avoid YAML:
            // containerTemplate {
            //     name 'shell'
            //     image 'ubuntu'
            //     command 'sleep'
            //     args 'infinity'
            // }
            cloud 'new_ci_idc'
            inheritFrom 'default'
            yaml '''
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: milvus-build
  namespace: jenkins
spec:
  nodeSelector:
    app: knowhere
  tolerations:
    - key: node-role.kubernetes.io/knowhere
      operator: Equal
      effect: NoSchedule
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - knowhere-e2e
          topologyKey: "kubernetes.io/hostname"
  enableServiceLinks: false
  containers:
  - name: main
    image: milvusdb/krte:20210722-806d13f
    env:
    - name: DOCKER_IN_DOCKER_ENABLED
      value: "true"
    - name: DOCKER_VOLUME_DIRECTORY
      value: "/mnt/disk/.docker"
    tty: true
    securityContext:
      privileged: true
    args: ["cat"]
    volumeMounts:
    - mountPath: /docker-graph
      name: docker-graph
    - mountPath: /var/lib/docker
      name: docker-root
    - mountPath: /lib/modules
      name: modules
      readOnly: true
    - mountPath: /sys/fs/cgroup
      name: cgroup
    - mountPath: /mnt/disk/.docker
      name: build-cache
      subPath: docker-volume
  volumes:
  - emptyDir: {}
    name: docker-graph
  - emptyDir: {}
    name: docker-root
  - hostPath:
      path: /tmp/krte/cache
      type: DirectoryOrCreate
    name: build-cache
  - hostPath:
      path: /lib/modules
      type: Directory
    name: modules
  - hostPath:
      path: /sys/fs/cgroup
      type: Directory
    name: cgroup
'''
            // Can also wrap individual steps:
            // container('shell') {
            //     sh 'hostname'
            // }
            defaultContainer 'main'
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
                        sh 'ls -lah'
                        sh 'chmod +x ci/docker/set_docker_mirror.sh'
                        sh './ci/docker/set_docker_mirror.sh'
                        def date = sh(returnStdout: true, script: 'date +%Y%m%d').trim()
                        def gitShortCommit = sh(returnStdout: true, script: 'git rev-parse --short HEAD').trim()
                        def image="milvusdb/knowhere-cpu-build:amd64-${os}-${date}-${gitShortCommit}"
                        sh "docker build -t ${image} -f ci/docker/builder/cpu/${params.os}/Dockerfile ."
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
