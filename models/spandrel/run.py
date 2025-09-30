import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import spandrel_extra_arches
import torch
from deepface import DeepFace
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
    glob_fn = input_dir.rglob if recursive else input_dir.glob
    return [
        p
        for p in glob_fn("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]


def get_transform(task: str) -> tv2.Compose:
    """Return the appropriate transform pipeline based on the task."""
    # For colorization, convert to grayscale; otherwise use RGB.
    if task == "color":

        def pre_convert(img):
            return img.convert("L")

        # Note: tv2.ToImage() expects a PIL image.
        return tv2.Compose(
            [
                tv2.Lambda(pre_convert),
                tv2.ToImage(),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
    else:

        def pre_convert(img):
            return img.convert("RGB")

        return tv2.Compose(
            [
                tv2.Lambda(pre_convert),
                tv2.ToImage(),
                tv2.ToDtype(torch.uint8, scale=True),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )


def process_and_return(
    img: Image.Image,
    model: ImageModelDescriptor,
    device: torch.device,
    transform: tv2.Compose,
) -> Image.Image:
    """Common processing: open image, apply transform, run through model, and return output."""
    with torch.inference_mode():
        image_tensor = transform(img).unsqueeze(0).to(device)
        output_tensor = model(image_tensor)
        output_tensor = output_tensor.squeeze(0).clamp(0, 1)
        # Post-process: convert tensor to uint8 then to PIL image.
        output_uint8 = tv2.ToDtype(torch.uint8, scale=True)(output_tensor)
        return tv2.ToPILImage()(output_uint8.cpu())


def process_face_image(
    image_input_path: Path,
    model: ImageModelDescriptor,
    device: torch.device,
) -> Image.Image | None:
    """Process faces in an image and merge enhanced faces back to original."""
    try:
        original_img = Image.open(image_input_path).convert("RGB")
        modified_img = original_img.copy()

        # Extract aligned faces using DeepFace
        faces = DeepFace.extract_faces(
            str(image_input_path),
            detector_backend="retinaface",
            align=True,
            grayscale=False,
        )

        # Re-use the general transform for face images.
        transform = get_transform(task="face")
        for face in faces:
            # Get face coordinates from original image.
            facial_area = face["facial_area"]
            x, y, w, h = (
                facial_area["x"],
                facial_area["y"],
                facial_area["w"],
                facial_area["h"],
            )

            # Process the aligned face.
            face_np = face["face"]  # numpy array (height, width, 3)

            # Convert to uint8 if necessary
            if face_np.dtype != "uint8":
                # If values are between 0 and 1, scale them to 0-255; otherwise, just cast.
                if face_np.max() <= 1.0:
                    face_np = (face_np * 255).astype("uint8")
                else:
                    face_np = face_np.astype("uint8")

            face_img = Image.fromarray(face_np)
            face_img = face_img.resize(
                (512, 512), Image.Resampling.LANCZOS
            )  # Resize to 512x512

            enhanced_face = process_and_return(face_img, model, device, transform)
            # Resize enhanced face to original face dimensions.
            enhanced_face = enhanced_face.resize((w, h), Image.Resampling.LANCZOS)
            modified_img.paste(enhanced_face, (x, y))

        return modified_img

    except ValueError as e:
        logger.warning(f"No faces detected in {image_input_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing faces in {image_input_path}: {str(e)}")
        raise


def process_image(
    image_input_path: Path,
    image_output_path: Path,
    model: ImageModelDescriptor,
    device: torch.device,
    task: str,
) -> None:
    """Process a single image based on the task and save the output."""
    try:
        if task == "face":
            output_image = process_face_image(image_input_path, model, device)
        else:
            transform = get_transform(task)
            img = Image.open(image_input_path)
            output_image = process_and_return(img, model, device, transform)

        if not output_image:
            raise

        output_image.save(image_output_path)
        logger.info(f"Saved processed image to: {image_output_path}")

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
