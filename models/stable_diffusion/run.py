import argparse
import os
import math
import random
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)


def compute_new_dim(dim: int, tile_size: int, overlap: int) -> int:
    """Compute a new dimension so that (dim - overlap) is an exact multiple of (tile_size - overlap).
    If the original dimension is smaller than the tile size, return tile_size (i.e. pad the image)."""
    step = tile_size - overlap
    if dim < tile_size:
        return tile_size
    n_steps = math.ceil((dim - overlap) / step)
    return n_steps * step + overlap


def split_image_overlap(image: Image.Image, tile_size: int, overlap: int) -> list:
    """
    Split an image into overlapping tiles.
    Each returned tile is exactly tile_size x tile_size.
    To ensure this, when near the right or bottom edge, the tile's top-left is shifted so that the crop
    is exactly tile_size x tile_size.
    Returns a list of tuples (tile, (left, upper, right, lower)).
    """
    width, height = image.size
    tiles = []
    step = tile_size - overlap
    y = 0
    while True:
        # Adjust y for bottom boundary
        if y + tile_size > height:
            y = height - tile_size
        x = 0
        while True:
            # Adjust x for right boundary
            if x + tile_size > width:
                x = width - tile_size
            box = (x, y, x + tile_size, y + tile_size)
            tile = image.crop(box)
            tiles.append((tile, box))
            if x + tile_size >= width:
                break
            x += step
        if y + tile_size >= height:
            break
        y += step
    return tiles


def create_weight_mask(tile_width: int, tile_height: int, overlap: int) -> np.ndarray:
    """
    Create a weight mask for blending a tile.
    The weight drops gradually near the edges over the overlap region.
    """
    mask = np.ones((tile_height, tile_width), dtype=np.float32)
    if overlap > 0:
        # Horizontal gradient
        grad_x = np.linspace(0, 1, min(overlap, tile_width), endpoint=False)
        for i in range(min(overlap, tile_width)):
            mask[:, i] *= grad_x[i]
            mask[:, -i - 1] *= grad_x[i]
        # Vertical gradient
        grad_y = np.linspace(0, 1, min(overlap, tile_height), endpoint=False)
        for j in range(min(overlap, tile_height)):
            mask[j, :] *= grad_y[j]
            mask[-j - 1, :] *= grad_y[j]
    return mask[..., None]


def merge_tiles(tiles: list, full_size: tuple, overlap: int) -> Image.Image:
    """
    Merge processed tiles using weighted blending.
    tiles is a list of tuples (processed_tile, (left, upper, right, lower)).
    full_size is the (width, height) of the full image.
    """
    merged = np.zeros((full_size[1], full_size[0], 3), dtype=np.float32)
    weight = np.zeros((full_size[1], full_size[0], 1), dtype=np.float32)

    for proc_tile, (left, upper, right, lower) in tiles:
        tile_np = np.array(proc_tile, dtype=np.float32)
        tile_h, tile_w = tile_np.shape[:2]
        mask = create_weight_mask(tile_w, tile_h, overlap)
        merged[upper:lower, left:right] += tile_np * mask
        weight[upper:lower, left:right] += mask

    blended = np.divide(merged, weight, where=weight > 1e-3)
    blended = np.uint8(np.clip(blended, 0, 255))
    return Image.fromarray(blended)


def process_image(image_path, args, pipe, generator):
    """
    Process a single image.
    If the image is square and exactly tile_size x tile_size, the pipeline is called directly.
    Otherwise, the image is resized to dimensions that are multiples of (tile_size - overlap) + overlap,
    split into overlapping tiles (each exactly tile_size x tile_size),
    processed tile‐by‐tile (using extra parameters for full tiles),
    and merged using weighted blending to reduce seams.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Skipping {image_path}: unable to read.")
        return None

    # Convert BGR to RGB and get original dimensions
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width, _ = original_img.shape

    tile_size = args.tile_size  # e.g., expected 1024
    overlap = args.overlap

    # Compute new dimensions that are multiples of (tile_size - overlap) plus overlap
    new_width = compute_new_dim(orig_width, tile_size, overlap)
    new_height = compute_new_dim(orig_height, tile_size, overlap)

    # Resize image to new dimensions using high-quality interpolation
    resized_np = cv2.resize(
        original_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )
    resized_image = Image.fromarray(resized_np)

    # If the resized image is exactly one tile, process it directly.
    if resized_image.size == (tile_size, tile_size):
        out = pipe(
            prompt=[args.prompt],
            negative_prompt=[args.negative_prompt],
            image=resized_image,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            width=tile_size,
            height=tile_size,
            num_inference_steps=args.num_inference_steps,
        )
        return out.images[0]

    # Otherwise, use overlapping tiling with fixed-size tiles.
    tiles = split_image_overlap(resized_image, tile_size, overlap)
    processed_tiles = []
    for tile, coords in tqdm(tiles, desc="Processing overlapping tiles"):
        w, h = tile.size
        # For full tiles, mimic the ControlNet tiling parameters.
        if (w, h) == (tile_size, tile_size):
            extra_kwargs = {
                "crops_coords_top_left": (tile_size, tile_size),
                "target_size": (tile_size, tile_size),
                "original_size": (tile_size * 2, tile_size * 2),
            }
        else:
            extra_kwargs = {}
        out = pipe(
            prompt=[args.prompt],
            negative_prompt=[args.negative_prompt],
            image=tile,
            control_image=tile,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
            width=w,
            height=h,
            num_inference_steps=args.num_inference_steps,
            **extra_kwargs,
        )
        processed_tiles.append((out.images[0], coords))
        if args.device == "cuda":
            torch.cuda.empty_cache()

    merged_image = merge_tiles(processed_tiles, resized_image.size, overlap)
    return merged_image


def main(args):
    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
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
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=torch_dtype,
        variant="fp16",
    )

    if args.device == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)

    # Use a random seed if provided seed is 0 or negative
    seed = args.seed if args.seed > 0 else random.randint(0, 2147483647)
    generator = torch.Generator(device=args.device).manual_seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f"restored_{filename}")
            if os.path.exists(output_path):
                print(f"Skipping {filename}: already processed.")
                continue
            try:
                restored_image = process_image(input_path, args, pipe, generator)
                if restored_image is not None:
                    restored_image.save(output_path)
                    print(f"Saved restored image to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restore images using ControlNet with optional tiling."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--output_dir", default="output", help="Path to the output directory."
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size in pixels (width and height).",
    )
    parser.add_argument(
        "--overlap", type=int, default=64, help="Overlap in pixels between tiles."
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
            "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
            "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        ),
        help="Prompt for restoration.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        ),
        help="Negative prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=30, help="Number of inference steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (use <=0 for random seed)."
    )

    args = parser.parse_args()
    main(args)
