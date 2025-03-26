#!/usr/bin/env bash

# sudo podman run --gpus all -it --entrypoint /bin/bash \
#  ddcolor-colorization

sudo podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  ddcolor-colorization \
  python run.py --model_path pretrain/ddcolor_modelscope.pth --input ./input --output ./output
