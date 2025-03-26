import argparse
import os
import random
from typing import Optional

import cv2
import numpy as np
from guided_filter import FastGuidedFilter
from tqdm import tqdm


def apply_gaussian_blur(
    image_np: np.ndarray, ksize: int = 5, sigmaX: float = 1.0
) -> np.ndarray:
    """Apply Gaussian blur with an odd kernel size."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)


def apply_guided_filter(
    image_np: np.ndarray, radius: int, eps: float, scale: float
) -> np.ndarray:
    """Apply a fast guided filter to enhance details and remove artifacts."""
    filter_obj = FastGuidedFilter(image_np, radius, eps, scale)
    return filter_obj.filter(image_np)


def convert_to_grayscale(image_np: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale."""
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image_np: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast.
    For grayscale images, histogram equalization is applied.
    For color images, CLAHE is applied on the L channel in LAB color space.
    """
    if len(image_np.shape) == 2:
        return cv2.equalizeHist(image_np)
    else:
        lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def denoise_image(image_np: np.ndarray) -> np.ndarray:
    """Denoise an image using fast non-local means denoising."""
    return cv2.fastNlMeansDenoisingColored(
        image_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )


def sharpen_image(image_np: np.ndarray) -> np.ndarray:
    """Enhance image sharpness using an unsharp mask approach."""
    gaussian = cv2.GaussianBlur(image_np, (0, 0), sigmaX=3)
    return cv2.addWeighted(image_np, 1.5, gaussian, -0.5, 0)


def resize_to_ratio(image_np: np.ndarray, ratio: float) -> np.ndarray:
    """
    Resize an image by the given ratio and adjust dimensions to be multiples of 8.

    Args:
        image_np: Input image as a numpy array.
        ratio: Scaling factor (e.g., 0.5 for 50% or 2.0 for 200%).

    Returns:
        Resized image as a numpy array.
    """
    height, width = image_np.shape[:2]
    new_width = max(8, int(round(width * ratio / 8)) * 8)
    new_height = max(8, int(round(height * ratio / 8)) * 8)
    return cv2.resize(
        image_np, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )


def process_image(image_path: str, args: argparse.Namespace) -> Optional[np.ndarray]:
    """Process a single image through the restoration pipeline."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Skipping {image_path}: unable to read.")
        return None

    # Ensure image is multiple of 8
    height, width = original_img.shape[:2]
    new_width = max(8, int(round(width / 8)) * 8)
    new_height = max(8, int(round(height / 8)) * 8)
    processed_np = cv2.resize(
        original_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Apply Gaussian blur if requested
    if args.apply_gaussian_blur:
        if args.blur_ksize is None:
            blur_strength = random.choice([i / 10.0 for i in range(10, 201, 2)])
            ksize = int(blur_strength)
            if ksize % 2 == 0:
                ksize += 1
            sigmaX = blur_strength / 2
        else:
            ksize = args.blur_ksize
            if ksize % 2 == 0:
                ksize += 1
            sigmaX = args.blur_sigma if args.blur_sigma is not None else ksize / 2
        processed_np = apply_gaussian_blur(processed_np, ksize=ksize, sigmaX=sigmaX)

    # Apply guided filter if requested
    if args.apply_guided_filter:
        radius = (
            args.gf_radius
            if args.gf_radius is not None
            else random.choice(range(1, 40, 2))
        )
        eps = (
            args.gf_eps
            if args.gf_eps is not None
            else random.choice([i / 1000.0 for i in range(1, 101, 2)])
        )
        scale = (
            args.gf_scale
            if args.gf_scale is not None
            else random.choice([i / 10.0 for i in range(10, 181, 5)])
        )
        processed_np = apply_guided_filter(processed_np, radius, eps, scale)

    # Denoise if requested
    if args.denoise:
        processed_np = denoise_image(processed_np)

    # Enhance contrast if requested
    if args.enhance_contrast:
        processed_np = enhance_contrast(processed_np)

    # Sharpen if requested
    if args.sharpen:
        processed_np = sharpen_image(processed_np)

    # Convert to grayscale if requested (this should be applied after contrast enhancement)
    if args.convert_grayscale:
        processed_np = convert_to_grayscale(processed_np)

    # Resize to a given ratio ensuring dimensions are multiples of 8.
    if args.resize_ratio != 1.0:
        processed_np = resize_to_ratio(processed_np, args.resize_ratio)

    return processed_np


def main(args: argparse.Namespace) -> None:
    """Set up the pipeline and process all images in the input directory."""
    os.makedirs(args.output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(args.input_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(args.input_dir, filename)
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(args.output_dir, f"processed_{base_name}.png")

            if os.path.exists(output_path):
                print(f"Skipping {filename}: already processed.")
                continue

            try:
                processed_image = process_image(input_path, args)
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)
                    print(f"Saved restored image to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance and restore images using OpenCV."
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
        "--apply_gaussian_blur",
        action="store_true",
        help="Apply Gaussian blur preprocessing.",
    )
    parser.add_argument(
        "--apply_guided_filter",
        action="store_true",
        help="Apply guided filter preprocessing.",
    )
    parser.add_argument(
        "--blur_ksize", type=int, help="Kernel size for Gaussian blur (must be odd)."
    )
    parser.add_argument("--blur_sigma", type=float, help="Sigma for Gaussian blur.")
    parser.add_argument("--gf_radius", type=int, help="Radius for guided filter.")
    parser.add_argument("--gf_eps", type=float, help="Epsilon for guided filter.")
    parser.add_argument(
        "--gf_scale", type=float, help="Scale factor for guided filter."
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Apply denoising to the image."
    )
    parser.add_argument(
        "--enhance_contrast", action="store_true", help="Enhance image contrast."
    )
    parser.add_argument("--sharpen", action="store_true", help="Sharpen the image.")
    parser.add_argument(
        "--convert_grayscale", action="store_true", help="Convert image to grayscale."
    )
    parser.add_argument(
        "--resize_ratio",
        type=float,
        default=1.0,
        help="Resize ratio (e.g., 0.5 for 50%, 2.0 for 200%)",
    )

    args = parser.parse_args()
    main(args)
