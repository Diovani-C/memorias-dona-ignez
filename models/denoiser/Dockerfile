FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends git ninja-build \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Ascend-Research/CascadedGaze.git /app

WORKDIR /app

# Use opencv headlees
RUN sed -i -e 's/opencv-python/opencv-python-headless/g' requirements.txt

# Install requirements
RUN pip install --upgrade pip && \
  pip install -r requirements.txt

RUN python setup.py develop --no_cuda_ext

RUN pip install gdown && \
  mkdir weights && \
  gdown --fuzzy 'https://drive.google.com/file/d/1Exf8LeYOjGnCLOvrA06P6UocFFmsunpQ/view?usp=sharing' -O weights/net_g_280000.pth 

# Copy your project files into the container
COPY . /app/
