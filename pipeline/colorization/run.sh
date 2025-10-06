#!/usr/bin/env bash

docker run --gpus all -it \
  -v ./output:/app/output:z \
  -v ./input:/app/input:z \
  ddcolor-colorization \
  python run.py --model_path pretrain/ddcolor_modelscope.pth --input ./input --output ./output
