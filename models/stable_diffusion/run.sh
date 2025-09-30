#!/usr/bin/env bash

# Enter container shell
# podman run --gpus all -it --entrypoint /bin/bash \
#   -v ./output:/workspace/output:z \
#   -v ./input:/workspace/input:z \
#  sd_tile

# Run container
podman run --gpus all -it \
  -v ./output:/workspace/output:z \
  -v ./input:/workspace/input:z \
  sd_tile \
  python run.py --input_dir ./input --prompt "A black and white photo of a family is posing for a picture outdoors, showcasing their clothing and smiles. The scene captures them with a backdrop of the sky and a vehicle nearby, adding a retro style to the moment. This image features multiple people including men and women standing together on the ground."
