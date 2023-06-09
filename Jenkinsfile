pipeline {
    agent any
    environment {
        registry = 'saadmakhdoom/real-time-automatic-attendance-system'
        dockerImage = ''
        DOCKERHUB_CREDENTIALS = credentials('saadmakhdoom-dockerhub')
    }
    stages {
        stage('Clone the repo') {
            steps {
                echo 'Clone the repo'
                sh 'rm -fr real-time-automatic-attendance-system'
                echo 'Get the repo'
                sh 'git clone https://github.com/MSaadMakhdoom/Facial-recognition-take-Automatic-Attendance-System.git'
                echo 'Set the dir'
                sh 'cd real-time-automatic-attendance-system'
                sh 'pwd'
            }
        }
        stage('Install Dependencies') {
            steps {
                echo 'Install dependencies'
                sh 'pwd'
                sh 'ls'
                dir('real-time-automatic-attendance-system') {
                    sh 'pip3 install -r requirements.txt'
                }
            }
        }
        
        stage('Build image, Push to Hub, Run the image') {
            steps {
                echo 'Docker image build '
                sh 'pwd'
                sh 'ls'
                script{
                    dir('real-time-automatic-attendance-system') {
                        dockerImage = docker.build("saadmakhdoom/real-time-automatic-attendance-system:latest")
                    }
                    if(dockerImage){
                        echo 'Docker Image built'
                        withDockerRegistry([credentialsId: "saadmakhdoom-dockerhub", url: ""]) {
                            dockerImage.push()
                        }
                        sh 'docker run -d -p 8000:8000 saadmakhdoom/real-time-automatic-attendance-system:latest'
                    }else{
                        error "Docker image build failed."
                    }
                }
            }
        }
        
    }
    post {
        always {
            sh 'docker logout'
        }
    }
}
