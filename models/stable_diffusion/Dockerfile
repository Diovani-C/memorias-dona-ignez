FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir -U pip && \
  pip install --no-cache-dir "diffusers[torch]" transformers accelerate opencv-python-headless pillow numpy tqdm

COPY download.py .

RUN python download.py

COPY . .
