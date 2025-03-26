FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends git wget python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/XPixelGroup/DiffBIR.git /app

WORKDIR /app

# Use opencv headlees
RUN sed -i -e 's/opencv_python/opencv-python-headless/g' requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

RUN mkdir weights
# --------------- stage-1 model weights ---------------
RUN wget https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth -O weights/BSRNet.pth
RUN wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt -O weights/face_swinir_v1.ckpt
RUN wget https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth -O weights/scunet_color_real_psnr.pth
RUN wget https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt -O weights/general_swinir_v1.ckpt
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth -O weights/realesrgan_s4_swinir_100k.pth

# --------------- pre-trained stable diffusion weights ---------------
RUN wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O weights/v2-1_512-ema-pruned.ckpt
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt -O weights/sd2.1-base-zsnr-laionaes5.ckpt

# --------------- IRControlNet weights ---------------
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth -O weights/v1_face.pth
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth -O weights/v1_general.pth
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth -O weights/v2.pth
RUN wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt -O weights/DiffBIR_v2.1.pt
