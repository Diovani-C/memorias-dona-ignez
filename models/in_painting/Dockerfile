FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
  pip install numpy torch blobfile tqdm pyYaml pillow

RUN git clone https://github.com/wyhuai/DDNM.git .

RUN curl https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt --create-dirs -o /workspace/exp/logs/imagenet/256x256_diffusion_uncond.pt 

# Copy your project files into the container
COPY . /workspace/

# Set the default command to run the export script.
# python main.py --ni --simplified --config oldphoto.yml --path_y oldphoto --eta 0.85 --deg "mask_color_sr" --deg_scale 2.0 --sigma_y 0.02 -i demo
CMD [ "python", "run.py" ]
