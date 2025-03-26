FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sczhou/CodeFormer.git /app

WORKDIR /app

# Use opencv headlees
RUN sed -i -e 's/opencv-python/opencv-python-headless/g' requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

RUN python basicsr/setup.py develop

RUN python scripts/download_pretrained_models.py facelib
RUN python scripts/download_pretrained_models.py CodeFormer
