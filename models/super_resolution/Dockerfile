FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git /app

WORKDIR /app

RUN sed -i -e 's/opencv-python/opencv-python-headless/g' requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

RUN python setup.py develop

# Download the pretrained models
RUN pip install gdown && \
  gdown https://drive.google.com/drive/folders/1D3BvTS1xBcaU1mp50k3pBzUWb7qjRvmB -O experiments/pretrained_models --folder 

COPY . /app/
