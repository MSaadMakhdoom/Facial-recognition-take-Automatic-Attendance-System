FROM python:3.8

WORKDIR /app
# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx

# Install OpenCV dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libavformat-dev \
    libpq-dev
    
# Install CMake
RUN apt-get update && apt-get install -y cmake
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
