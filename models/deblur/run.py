import argparse
import os
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from basicsr.metrics import calculate_ssim
from basicsr.models.archs.AdaRevID_arch import AdaRevIDSlide as Net
from natsort import natsorted
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from tqdm import tqdm

import utils  # Assumes utils has load_img and save_img functions


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Single Image Motion Deblurring using Restormer"
    )
    parser.add_argument(
        "--input_dir",
        default="/app/input/",
        type=str,
        help="Directory of validation images",
    )
    parser.add_argument(
        "--tar_dir",
        default="/data/mxt_data/GoPro/test/sharp",
        type=str,
        help="Directory of ground truth images",
    )
    parser.add_argument(
        "--result_dir",
        default="/app/output/",
        type=str,
        help="Directory for saving results",
    )
    parser.add_argument(
        "--weights",
        default="/app/weights/RevD-B_GoPro.pth",
        type=str,
        help="Path to model weights",
    )
    parser.add_argument(
        "--dataset", default="GoPro", type=str, help="Test dataset name"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        type=bool,
        help="Flag to save the output results",
    )
    parser.add_argument(
        "--get_psnr",
        default=False,
        type=bool,
        help="Calculate PSNR metric",
    )
    parser.add_argument(
        "--get_ssim",
        default=False,
        type=bool,
        help="Calculate SSIM metric",
    )
    parser.add_argument(
        "--gpus", default="0", type=str, help="GPU IDs to use (if applicable)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force model to run on CPU instead of GPU",
    )
    parser.add_argument(
        "--yaml_file",
        default="AdaRevID-B-GoPro.yml",
        type=str,
        help="Path to the yaml config file",
    )

    return parser.parse_args()


def load_configuration(yaml_file):
    """Loads the YAML configuration file."""
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(yaml_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def load_model(config, weights, device):
    """
    Loads and initializes the model with the provided configuration and weights.

    Args:
        config (dict): Model configuration loaded from YAML.
        weights (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.

    Returns:
        nn.Module: The initialized restoration model.
    """
    network_config = config["network_g"]
    # Remove the 'type' key if present.
    network_config.pop("type", None)
    model = Net(**network_config)

    if weights is not None:
        checkpoint = torch.load(weights, map_location=device)
        try:
            model.load_state_dict(checkpoint["params"], strict=False)
        except KeyError:
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except KeyError:
                # Remove potential 'module.' prefix if model was saved with DataParallel.
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith("module.") else k
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)
    return model


def process_images(model, args, device):
    """
    Processes each image: runs inference, calculates metrics, and optionally saves the output.

    Args:
        model (nn.Module): The restoration model.
        args: Parsed command-line arguments.
        device (torch.device): Device for inference.
    """
    psnr_values = []
    ssim_values = []
    total_time = 0.0
    last_num_decoders = None

    if args.save_result:
        os.makedirs(args.result_dir, exist_ok=True)

    # Get sorted list of image files.
    image_files = natsorted(
        glob(os.path.join(args.input_dir, "*.png"))
        + glob(os.path.join(args.input_dir, "*.jpg"))
    )

    with torch.inference_mode():
        for file_path in tqdm(image_files, desc="Processing images"):
            if device.type == "cuda":
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            # Load and preprocess image.
            img = np.float32(utils.load_img(file_path)) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()

            # Run inference. If the model returns a tuple, assume it is (restored, num_decoders).
            output = model(img_tensor)
            if isinstance(output, (tuple, list)) and len(output) == 2:
                restored, num_decoders = output
                last_num_decoders = num_decoders
            else:
                restored = output

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            total_time += elapsed

            # Handle different possible output formats.
            if isinstance(restored, list):
                restored = restored[-1]
            elif isinstance(restored, dict):
                restored = restored.get("img", restored)
                if isinstance(restored, list):
                    restored = restored[-1]

            restored = torch.clamp(restored, 0, 1).cpu().detach()
            restored_img = restored.squeeze(0).permute(1, 2, 0).numpy()
            restored_img_uint8 = img_as_ubyte(restored_img)

            # If ground truth is available, calculate metrics.
            if args.get_psnr or args.get_ssim:
                gt_file = os.path.join(args.tar_dir, os.path.basename(file_path))
                if not os.path.exists(gt_file):
                    gt_file = gt_file[:-3] + "png"  # try alternative extension
                if os.path.exists(gt_file):
                    gt_img = cv2.imread(gt_file)
                    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                    if args.get_psnr:
                        psnr_val = psnr_loss(restored_img_uint8, gt_img)
                        psnr_values.append(psnr_val)
                    if args.get_ssim:
                        ssim_val = calculate_ssim(
                            restored_img_uint8, gt_img, crop_border=0
                        )
                        ssim_values.append(ssim_val)

            # Optionally save the restored image.
            if args.save_result:
                result_path = os.path.join(
                    args.result_dir,
                    os.path.splitext(os.path.basename(file_path))[0] + ".png",
                )
                utils.save_img(result_path, restored_img_uint8)

    avg_time = total_time / len(image_files) if image_files else 0
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0

    print(f"Average inference time per image: {avg_time:.4f} seconds")
    if args.get_psnr:
        print(f"Average PSNR: {avg_psnr:.4f}")
    if args.get_ssim:
        print(f"Average SSIM: {avg_ssim:.4f}")

    if last_num_decoders is not None:
        print(f"num_decoders: {last_num_decoders}")


def main():
    args = parse_args()
    # Choose device: CPU if --cpu is set or if CUDA is unavailable.
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Using device: {device}")

    # Load YAML configuration.
    config = load_configuration(args.yaml_file)

    # Load model and move it to the selected device.
    model = load_model(config, args.weights, device)
    model.to(device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.eval()

    print("===> Testing using weights:", args.weights)
    process_images(model, args, device)


if __name__ == "__main__":
    main()
