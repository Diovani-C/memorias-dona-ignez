FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
  pip install spandrel spandrel_extra_arches pillow tqdm gdown

# Download the pretrained models
RUN mkdir weights

# Real DRCT-L 4x image super resolution model
RUN gdown --fuzzy https://drive.google.com/file/d/13P6X5gzNAD_9pMI-Cb-3MY3CdARDBgrp/view?usp=drive_link -O weights/real_drct-l_4x.pth
# DDColor colorization model 
RUN wget --no-verbose --show-progress --progress=bar:force:noscroll https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth -O weights/ddcolor_modelscope.pth 
# codeformer face restoration model
RUN wget --no-verbose --show-progress --progress=bar:force:noscroll https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O weights/codeformer.pth
# MIRNetv2 denoise model
RUN wget --no-verbose --show-progress --progress=bar:force:noscroll https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth -O weights/real_denoising.pth
# MIRNetv2 low light image enhancer model
RUN wget --no-verbose --show-progress --progress=bar:force:noscroll https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth -O weights/enhancement_lol.pth
# FFTformer deblurring model 
RUN wget --no-verbose --show-progress --progress=bar:force:noscroll https://github.com/kkkls/FFTformer/releases/download/pretrain_model/fftformer_GoPro.pth -O weights/fftformer_GoPro.pth

COPY . /app/
