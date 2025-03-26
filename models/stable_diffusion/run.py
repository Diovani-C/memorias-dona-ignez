import argparse
import os
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


def split_image(image: Image.Image, tile_size: int, overlap: int) -> list:
    """Split image into overlapping tiles, resizing if the original image is smaller than the tile size."""
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and less than tile_size.")

    original_width, original_height = image.size
    scaling_factor = 1.0

    # Scale up the image if its largest dimension is smaller than the tile size
    if max(original_width, original_height) < tile_size:
        scaling_factor = tile_size / max(original_width, original_height)
        new_width = int(round(original_width * scaling_factor))
        new_height = int(round(original_height * scaling_factor))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    width, height = image.size
    tiles = []
    step = tile_size - overlap

    # Iterate over the image with the defined step to create overlapping tiles
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Calculate the bounding box for the current tile
            left = x
            upper = y
            right = min(x + tile_size, width)
            lower = min(y + tile_size, height)

            tile = image.crop((left, upper, right, lower))

            # If the tile is smaller than tile_size, pad it with black pixels (RGB: 0, 0, 0)
            if tile.size != (tile_size, tile_size):
                padded_tile = Image.new("RGB", (tile_size, tile_size))
                padded_tile.paste(tile, (0, 0))
                tile = padded_tile

            # Calculate the corresponding original coordinates (if the image was resized)
            if scaling_factor != 1.0:
                orig_left = round(left / scaling_factor)
                orig_upper = round(upper / scaling_factor)
                orig_right = round(right / scaling_factor)
                orig_lower = round(lower / scaling_factor)

                # Clamp the coordinates to the original image dimensions
                orig_left = max(0, min(orig_left, original_width))
                orig_upper = max(0, min(orig_upper, original_height))
                orig_right = max(0, min(orig_right, original_width))
                orig_lower = max(0, min(orig_lower, original_height))
                original_coords = (orig_left, orig_upper, orig_right, orig_lower)
            else:
                original_coords = (left, upper, right, lower)

            tiles.append((tile, original_coords))

    return tiles


def create_weight_mask(tile_size: int, overlap: int) -> np.ndarray:
    """Create a weight mask for blending tiles."""
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    gradient = np.linspace(0, 1, overlap, endpoint=False)
    for i in range(overlap):
        mask[:, i] *= gradient[i]
        mask[:, -i - 1] *= gradient[i]
        mask[i, :] *= gradient[i]
        mask[-i - 1, :] *= gradient[i]
    return mask[..., None]


def merge_tiles(tiles: list, full_size: tuple, overlap: int) -> Image.Image:
    """Merge processed tiles into the final image."""
    merged = np.zeros((full_size[1], full_size[0], 3), dtype=np.float32)
    weight = np.zeros((full_size[1], full_size[0], 1), dtype=np.float32)
    mask = create_weight_mask(tiles[0][0].size[0], overlap)
    for processed_tile, (left, upper, right, lower) in tiles:
        tile_width = right - left
        tile_height = lower - upper
        tile_array = np.array(processed_tile, dtype=np.float32)
        valid_region = tile_array[:tile_height, :tile_width]
        valid_mask = mask[:tile_height, :tile_width]
        merged[upper:lower, left:right] += valid_region * valid_mask
        weight[upper:lower, left:right] += valid_mask
    merged = np.divide(merged, weight, where=weight > 1e-3)
    return Image.fromarray(np.uint8(np.clip(merged, 0, 255)))


def process_image(image_path, args, pipe, generator):
    """Process a single image through the restoration pipeline."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Skipping {image_path}: unable to read.")
        return None

    # Make the image multiple of 64
    height, width, _ = original_img.shape
    new_width = int(round(width / 64) * 64)
    new_height = int(round(height / 64) * 64)
    processed_np = cv2.resize(
        original_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )

    processed_np = cv2.cvtColor(processed_np, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_np)

    # Split into tiles
    tiles = split_image(processed_image, args.tile_size, args.overlap)
    processed_tiles = []

    for tile, coords in tqdm(tiles, desc="Processing tiles"):
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=tile,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            width=args.tile_size,
            height=args.tile_size,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            guidance_scale=0,
        )
        processed_tiles.append((output.images[0], coords))
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Merge tiles
    merged_image = merge_tiles(processed_tiles, processed_image.size, args.overlap)
    return merged_image


def main(args):
    """Main function to set up the pipeline and process images."""
    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32
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

    if args.device == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)

    # Seed generator for reproducibility
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image in the input directory
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
    parser = argparse.ArgumentParser(description="Restore old photos using ControlNet.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--output_dir", default="output", help="Path to the output directory."
    )
    parser.add_argument(
        "--tile_size", type=int, default=1024, help="Size of each tile."
    )
    parser.add_argument(
        "--overlap", type=int, default=64, help="Overlap between tiles."
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
        default="Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.",
        help="Prompt for restoration.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth.",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    main(args)
