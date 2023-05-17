FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
