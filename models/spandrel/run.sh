#!/usr/bin/env bash

# sudo podman run --gpus all -it --entrypoint /bin/bash \
# spandrel

podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  spandrel \
  python run.py -i input -o output --task deblur --cpu
