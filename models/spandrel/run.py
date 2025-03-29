import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import spandrel_extra_arches
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
from torchvision.transforms import v2 as tv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-trained model paths for each task (relative to script directory)
MODEL_PATHS = {
    "upscale_4x": "weights/real_drct-l_4x.pth",
    "color": "weights/ddcolor_modelscope.pth",
    "face": "weights/codeformer.pth",
    "denoise": "weights/real_denoising.pth",
    "deblur": "weights/fftformer_GoPro.pth",
    "low_light": "weights/enhancement_lol.pth",
}

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Image processing using Spandrel models"
    )
    parser.add_argument(
        "-i",
        "--in_path",
        type=str,
        required=True,
        help="Input image or directory path.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        required=True,
        help="Output directory path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(MODEL_PATHS.keys()),
        help="Task for the model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the use of CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images recursively in subdirectories.",
    )
    return parser.parse_args()


def validate_paths(in_path: Path, out_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate input and output paths."""
    if not in_path.exists():
        return False, f"Input path does not exist: {in_path}"
    if out_path.exists() and not out_path.is_dir():
        return False, f"Output path exists and is not a directory: {out_path}"
    return True, None


def get_image_paths(input_dir: Path, recursive: bool) -> List[Path]:
    """Get list of image paths from input directory."""
    if recursive:
        return [
            p
            for p in input_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        ]
    return [
        p
        for p in input_dir.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]


def process_image(
    image_input_path: Path,
    image_output_path: Path,
    model: ImageModelDescriptor,
    device: torch.device,
    task: str,
) -> None:
    """Process a single image through the model and save the output."""
    try:
        with torch.inference_mode(), Image.open(image_input_path) as img:
            # Convert to grayscale for colorization task
            if task == "color":
                # DDColor expects single channel input
                image = img.convert("L")
            else:
                image = img.convert("RGB")

            # Create transform pipeline
            transform = tv2.Compose(
                [
                    tv2.ToImage(),
                    tv2.ToDtype(torch.uint8, scale=True),
                    tv2.ToDtype(torch.float32, scale=True),
                ]
            )

            # Process image tensor
            image_tensor = transform(image)

            # Add batch dimension and ensure correct shape
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Model inference
            output_tensor = model(image_tensor)

            # Post-process output
            output_tensor = output_tensor.squeeze(0).clamp(0, 1)
            output_uint8 = tv2.ToDtype(torch.uint8, scale=True)(output_tensor)
            output_image = tv2.ToPILImage()(output_uint8.cpu())

            # Ensure output directory exists
            image_output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(image_output_path)

    except Exception as e:
        logger.error(f"Failed to process {image_input_path}: {str(e)}")
        raise


def load_model(model_path: Path, device: torch.device) -> ImageModelDescriptor:
    """Load and prepare the specified model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spandrel_extra_arches.install()
    model = ModelLoader().load_from_file(str(model_path))

    if not isinstance(model, ImageModelDescriptor):
        raise ValueError("Loaded model is not a valid ImageModelDescriptor")

    return model.to(device).eval()


def main() -> None:
    """Main processing function."""
    args = parse_arguments()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()

    # Validate paths
    valid, msg = validate_paths(in_path, out_path)
    if not valid:
        raise ValueError(msg)

    # Prepare device
    device = torch.device(
        "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Load model
    model_path = script_dir / MODEL_PATHS[args.task]
    model = load_model(model_path, device)

    # Process input
    if in_path.is_file():
        if in_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {in_path.suffix}")

        output_path = out_path / f"{in_path.stem}.png"
        process_image(in_path, output_path, model, device, args.task)
        logger.info(f"Processed image saved to: {output_path}")

    elif in_path.is_dir():
        image_paths = get_image_paths(in_path, args.recursive)
        if not image_paths:
            raise ValueError(f"No valid images found in {in_path}")

        for img_path in tqdm(image_paths, desc="Processing images"):
            rel_path = img_path.relative_to(in_path)
            output_path = out_path / rel_path.with_name(f"{rel_path.stem}.png")
            try:
                process_image(img_path, output_path, model, device, args.task)
            except Exception:
                logger.error(f"Skipping {img_path} due to processing error")

    else:
        raise ValueError(f"Invalid input path: {in_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise
