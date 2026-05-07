pipeline {
    agent none

    stages {
        stage("Disabled"){
            steps {
                echo "SSE CI is disabled."
            }
        }
    }
}
