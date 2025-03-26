import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)

# File used to download all the models in the docker build step
torch_dtype = torch.float16
base_model = "stabilityai/stable-diffusion-xl-base-1.0"

eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    base_model, subfolder="scheduler"
)

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-tile-sdxl-1.0",
    use_safetensors=True,
    torch_dtype=torch_dtype,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    use_safetensors=True,
    torch_dtype=torch_dtype,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch_dtype,
    scheduler=eulera_scheduler,
    variant="fp16",
)
