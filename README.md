[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
This project utilizes  Python programing for building the application

[![Docker](https://img.shields.io/badge/Docker-Containerization-blue)](https://www.docker.com/)
This project utilizes Docker for containerizing .

[![AWS ECS](https://img.shields.io/badge/AWS-ECS-orange)](https://aws.amazon.com/ecs/)
This project utilizes AWS ECS (Elastic Container Service) for running containers.

[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-orange)](https://dvc.org/)
This project utilizes DVC (Data Version Control) for managing and versioning data sets.




# Facial-recognition-take-Automatic-Attendance-System
The Real-Time Automatic Attendance System is a project aimed at improving attendance tracking and management in educational institutions, offices, and other organizations. The system utilizes modern technologies such as computer vision and machine learning to provide real-time tracking of attendance and automatically generate attendance reports.
##
[![Python application](https://github.com/MSaadMakhdoom/Facial-recognition-take-Automatic-Attendance-System/actions/workflows/python-app.yml/badge.svg)](https://github.com/MSaadMakhdoom/Facial-recognition-take-Automatic-Attendance-System/actions/workflows/python-app.yml)
##
[![Docker-image-real-time-automatic-attendance-system](https://github.com/MSaadMakhdoom/Facial-recognition-take-Automatic-Attendance-System/actions/workflows/docker-image-app.yml/badge.svg)](https://github.com/MSaadMakhdoom/Facial-recognition-take-Automatic-Attendance-System/actions/workflows/docker-image-app.yml)
## 
Docker link : https://hub.docker.com/repository/docker/saadmakhdoom/real-time-automatic-attendance-system/
## Functional Requirements for Real-Time Face Recognition Automatic Attendance Management System:
1. Image Preprocessing: The system should perform necessary preprocessing on the
captured images or video frames, such as resizing, normalization, and noise
reduction, to enhance the quality and accuracy of face detection and recognition.
2. Face Detection: The system should be able to detect human faces in real-time from
a live video stream or images with high accuracy and reliability, even in varying
lighting conditions and orientations.
3. Face Recognition: The system should be able to recognize and identify faces.
4. Attendance Tracking: The system should be able to automatically track attendance
by matching detected faces with the known faces in the database.
5. Database Management: The system should allow for easy management of a
database of known faces, including the ability to add, update, and delete faces from
the database.
6. Real-Time Processing: The system should operate in real-time, with minimal delay
between face detection, recognition, and attendance logging.
7. User Interface: The system should have a user-friendly interface that allows
administrators to easily configure and manage the system, including adding and
updating faces in the database
8. Reporting: The system should generate comprehensive attendance reports.
9. User Authentication: The system should have robust user authentication
mechanisms to ensure that only authorized personnel can access the attendance
data.
10. User-Friendly Interface: The system should have a user-friendly interface.


# Setup Instructions

## Virtual Environment

Create a virtual environment:

```
python3 -m venv venv
```

Activate the virtual environment:

```
source venv/bin/activate
```

Install the required packages from `requirements.txt`:

```
pip install -r requirements.txt
```

## Database Schema

Apply the necessary changes to the database schema:

```
python manage.py makemigrations
python manage.py migrate
```

## Django Application Execution

Start the Django application:

```
python manage.py runserver
```

## DVC Setup

Install DVC and the S3 bucket remote storage:

```
pip install dvc-s3
```

Set the DVC storage name:

```
dvc remote add -d img s3://projectmlops/
```

Add the dataset images to the remote folder:

```
dvc add ./img
```

Pull data from the S3 bucket:

```
aws configure
dvc pull
```

Create and run the DVC pipeline:

```
dvc run -n model_train -d face_detection_model_svm.py -o confusion_matrix.png --no-exec python3 face_detection_model_svm.py
dvc repro
```

## Airflow Setup

Install Apache Airflow:

```
pip install apache-airflow

pip install apache-airflow-providers-cncf-kubernetes
```


Create an Airflow directory:

```
mkdir Airflow
export AIRFLOW_HOME=.
```

Initialize the Airflow database:

```
airflow db init
```

Create an admin account:

```
airflow users create --username msaad --firstname Muhammad --lastname Saad --email msaadmakhdoom@gmail.com --role Admin --password 123456
```

Start the Airflow web server:

```
airflow webserver -p 8080
```
