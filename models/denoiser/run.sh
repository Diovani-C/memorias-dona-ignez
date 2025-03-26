#!/usr/bin/env bash

# Enter container shell
# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/workspace/output:z \
#   -v ./input:/workspace/input:z \
#  cascade_gaze_denoise

# Run container
sudo podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  cascade_gaze_denoise \
  python run.py -opt CascadedGaze-SIDD.yml # --use_cpu
