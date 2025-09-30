#!/usr/bin/env bash

# Enter container shell
# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/app/output:z \
#   -v ./input:/app/input:z \
#  image_processing

# Run container
podman run -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  image_processing \
  python run.py --input_dir ./input --resize_ratio 2.0
# python run.py --input_dir ./input --apply_gaussian_blur --apply_guided_filter --convert_grayscale --sharpen --enhance_contrast --denoise
