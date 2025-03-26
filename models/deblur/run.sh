#!/usr/bin/env bash

# Enter container shell
# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/app/output:z \
#   -v ./input:/app/input:z \
#   adarevd-deblur

sudo podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  adarevd-deblur \
  python run.py --cpu
