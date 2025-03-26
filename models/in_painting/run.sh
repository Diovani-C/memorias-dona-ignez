#!/usr/bin/env bash

sudo podman run --device nvidia.com/gpu=all \
  -v ~/Projects/memorias-dona-ines/output:/app/output:z \
  -v ~/Projects/memorias-dona-ines/input:/app/input:z

sudo podman run --device nvidia.com/gpu=all -it --entrypoint /bin/bash \
  -v ~/Projects/memorias-dona-ines/output:/workspace/output:z \
  -v ~/Projects/memorias-dona-ines/input:/workspace/input:z \
  43a0cf1ccd8c
