FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git curl g++ ninja-build unzip \
  && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/INVOKERer/AdaRevD.git /app

WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip && \
  pip install matplotlib scikit-learn scikit-image opencv-python-headless yacs joblib natsort h5py tqdm ptflops \
  seaborn spectral einops kornia timm gdown addict future lmdb numpy pyyaml requests scipy tb-nightly lpips

# Install basicsr in development mode
RUN python setup.py develop --no_cuda_ext --record install_files.txt

# Download models weights
RUN pip install gdown && \
  gdown https://drive.google.com/drive/folders/1r4SKc6jRBK5gr1Kl4iORHBjd-5TRKTSY -O weights --folder

RUN unzip weights/classifier.zip -d weights
RUN rm weights/classifier.zip

# Copy your project files into the container
COPY . /app/Motion_Deblurring/
WORKDIR /app/Motion_Deblurring
