FROM python:bookworm

RUN pip install --no-cache-dir -U pip && \
  pip install --no-cache-dir opencv-python-headless numpy tqdm

COPY . /app

WORKDIR /app
