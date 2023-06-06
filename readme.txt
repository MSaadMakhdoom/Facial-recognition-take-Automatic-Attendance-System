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