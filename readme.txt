# create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate
# install requirements.txt
pip install -r requirements.txt

# changes needed to bring the database schema
python manage.py makemigrations
# apply any pending migrations to your database.
python manage.py migrate

# django application execution
python manage.py runserver

# dvc
# remote stroage s3 bucket

# install 
pip install dvc-s3

# dvc stroage name 
img



dvc remote add -d img s3://projectmlops/

# Add images to remote folder
dvc add ./img


# dvc pull data s3 bucket
aws configure

dvc pull


# DVC pipeline
dvc run -n model_train -d face_detection_model_svm.py -o confusion_matrix.png --no-exec python3 face_detection_model_svm.py                     