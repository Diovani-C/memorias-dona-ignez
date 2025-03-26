#!/usr/bin/env bash

# Enter container shell
# sudo podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/app/output:z \
#   -v ./input:/app/input:z \
#   face-restoration_codeformer

sudo podman run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  face-restoration_codeformer \
  python inference_codeformer.py -w 0.7 --input_path input --output_path output --bg_upsampler realesrgan --face_upsample
