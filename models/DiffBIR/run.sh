#!/usr/bin/env bash

sudo podman run --gpus all -it --entrypoint /bin/bash \
  diffbir

# -v ./output:/workspace/output:z \
# -v ./input:/workspace/input:z \
sudo podman run --gpus all -it \
  diffbir \
  python3 -u inference.py \
  --task denoise \
  --upscale 1 \
  --version v2.1 \
  --device cpu \
  --precision fp32 \
  --captioner none \
  --cfg_scale 8 \
  --noise_aug 0 \
  --input inputs/demo/bid \
  --output results/v2.1_demo_bid
# --cleaner_tiled \
# --vae_encoder_tiled \
# --vae_decoder_tiled \
# --cldm_tiled
