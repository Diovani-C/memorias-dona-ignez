FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git wget build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/piddnad/DDColor.git /app

WORKDIR /app

# Use opencv headless because of the system dependencies
RUN sed -i -e 's/opencv_python/opencv-python-headless/g' requirements.txt
# Update the torch version
RUN sed -i -e 's/torch>=1.7/torch>=2.2/g' requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt 

RUN wget https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth -O pretrain/ddcolor_modelscope.pth

COPY . /app/
