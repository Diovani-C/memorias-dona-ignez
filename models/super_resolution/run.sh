#!/usr/bin/env bash

# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   atd-super-resolution

sudo podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  atd-super-resolution \
  python run.py -i input/ -o output/ --scale 2 --task classical --cpu
