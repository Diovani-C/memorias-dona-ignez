#!/usr/bin/env bash

# Enter container shell
# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/workspace/output:z \
#   -v ./input:/workspace/input:z \
#  cascade_gaze_denoise

# Run container
sudo podman run --gpus all -it \
  -v ./output:/workspace/output:z \
  -v ./input:/workspace/input:z \
  sd_tile \
  python run.py --input_dir ./input
