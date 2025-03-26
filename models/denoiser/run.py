# ------------------------------------------------------------------------
# Modified from CascadedGaze (https://github.com/Ascend-Research/CascadedGaze)
# ------------------------------------------------------------------------
import argparse
import logging
import os
from typing import Any

import torch
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, imwrite, tensor2img
from basicsr.utils.options import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_options_inference() -> dict:
    """
    Minimal options parser for inference only.
    Parses a YAML file and optional command-line overrides for image paths.
    """
    parser = argparse.ArgumentParser(description="Inference options")
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="Path to input image directory")
    parser.add_argument(
        "--output_path", type=str, help="Path to output image directory"
    )
    parser.add_argument("--cpu", action="store_true", help="Run the model on CPU")
    args = parser.parse_args()

    # Parse the YAML configuration; set is_train to False for inference.
    opt = parse(args.opt, is_train=False)

    # Override image paths if provided via command line.
    if args.input_path and args.output_path:
        # Use keys matching your inference code (e.g. input_dir/output_dir)
        opt["img_path"] = {"input_dir": args.input_path, "output_dir": args.output_path}

    # Disable distributed processing for inference.
    opt["dist"] = False

    return opt


def process_image(
    file_client: FileClient, model: Any, input_path: str, output_path: str
) -> None:
    """
    Process a single image using the provided model.

    Args:
        file_client (FileClient): The file client used for reading images.
        model (Any): The pre-loaded super-resolution model.
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
    """
    try:
        # Read image bytes from file
        img_bytes = file_client.get(input_path, None)
    except Exception as e:
        logging.error(f"Failed to read file {input_path}: {e}")
        return

    try:
        # Convert bytes into an image with float32 precision
        img = imfrombytes(img_bytes, float32=True)
    except Exception as e:
        logging.error(
            f"Error converting bytes to image for {input_path}: {e}. Skipping this file."
        )
        return

    # Convert image to tensor with proper formatting and add a batch dimension
    img_tensor = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0)

    # Feed image data into the model
    model.feed_data(data={"lq": img_tensor})

    # Apply grids if the model validation options require it
    if model.opt.get("val", {}).get("grids", False):
        model.grids()

    model.test()

    if model.opt.get("val", {}).get("grids", False):
        model.grids_inverse()

    # Retrieve the output and convert tensor to image format
    visuals = model.get_current_visuals()
    sr_img = tensor2img(visuals["result"])

    # Write the processed image to the output path
    imwrite(sr_img, output_path)
    logging.info(f"Processed {os.path.basename(input_path)} -> {output_path}")


def main() -> None:
    """
    Main function to perform super-resolution on images from the input directory.

    Command-line Args:
        --cpu: If specified, the model will run on the CPU.
    """
    # Setup command-line arguments
    opt = parse_options_inference()

    # Parse configuration options
    if opt.get("cpu", False):
        logging.info("Running on CPU as specified by the command-line argument.")
        opt["num_gpu"] = 0
    else:
        opt["num_gpu"] = torch.cuda.device_count()
        logging.info(f"Detected {opt['num_gpu']} GPU(s).")

    # Retrieve input and output directory paths from the options
    img_path_options = opt.get("img_path", {})
    input_dir: str = img_path_options.get("input_dir")
    output_dir: str = img_path_options.get("output_dir")

    if not input_dir or not output_dir:
        raise ValueError(
            "Both 'input_dir' and 'output_dir' must be specified in opt['img_path']"
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the file client and model
    file_client = FileClient("disk")
    opt["dist"] = False
    model = create_model(opt)

    # Supported image extensions
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Process each image in the input directory using torch inference mode
    with torch.inference_mode():
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(supported_extensions):
                input_path = os.path.join(input_dir, filename)
                # Change output filename to PNG regardless of input format
                base_name = os.path.splitext(filename)[0]
                out_file = os.path.join(output_dir, base_name + ".png")
                process_image(file_client, model, input_path, out_file)


if __name__ == "__main__":
    main()
