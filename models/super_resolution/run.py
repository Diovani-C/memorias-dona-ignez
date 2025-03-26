import argparse
import os
import os.path as osp

import torch
from PIL import Image
from torchvision import transforms

from basicsr.archs.atd_arch import ATD

# Pretrained model paths for each task and scale
MODEL_PATHS = {
    "classical": {
        "2": "experiments/pretrained_models/001_ATD_SRx2_finetune.pth",
        "3": "experiments/pretrained_models/002_ATD_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/003_ATD_SRx4_finetune.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/101_ATD_light_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/102_ATD_light_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/103_ATD_light_SRx4_finetune.pth",
    },
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ATD Super-Resolution Inference")
    parser.add_argument(
        "-i",
        "--in_path",
        type=str,
        default="input/",
        help="Input image or directory path.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="output/",
        help="Output directory path.",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
        "--task",
        type=str,
        default="classical",
        choices=["classical", "lightweight"],
        help=(
            "Task for the model. 'classical' for classical SR models and "
            "'lightweight' for lightweight models."
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="Force the use of CPU.")
    parser.add_argument(
        "--compile", action="store_true", help="Apply torch.compile to the model."
    )
    return parser.parse_args()


def process_image(image_input_path, image_output_path, model, device):
    """
    Process a single image through the model and save the output.

    Args:
        image_input_path (str): Path to the input image.
        image_output_path (str): Path where the output image will be saved.
        model (torch.nn.Module): The super-resolution model.
        device (str): Device to run inference on.
    """
    with torch.inference_mode():
        # Load and preprocess the input image.
        image = Image.open(image_input_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        # Run the model and clamp the output.
        output_tensor = model(image_tensor).clamp(0.0, 1.0)[0].cpu()

        # Convert the tensor to a PIL image and save it.
        output_image = transforms.ToPILImage()(output_tensor)
        output_image.save(image_output_path)


def main():
    args = parse_arguments()

    # Select device: GPU if available and not forced to use CPU.
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    # Set model hyperparameters based on the task.
    if args.task == "lightweight":
        embed_dim = 48
        depths = [6, 6, 6, 6]
        num_heads = [4, 4, 4, 4]
        category_size = 128
        num_tokens = 64
        reducted_dim = 8
        convffn_kernel_size = 7
        mlp_ratio = 1
        upsampler = "pixelshuffledirect"
    else:
        embed_dim = 210
        depths = [6, 6, 6, 6, 6, 6]
        num_heads = [6, 6, 6, 6, 6, 6]
        category_size = 256
        num_tokens = 128
        reducted_dim = 20
        convffn_kernel_size = 5
        mlp_ratio = 2
        upsampler = "pixelshuffle"

    # Initialize the model with the specified parameters.
    model = ATD(
        upscale=args.scale,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=16,
        category_size=category_size,
        num_tokens=num_tokens,
        reducted_dim=reducted_dim,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        use_checkpoint=False,
    )

    # Load the pretrained model weights.
    model_file = MODEL_PATHS[args.task][str(args.scale)]
    checkpoint = torch.load(model_file, map_location=device)
    state_dict = checkpoint["params_ema"]
    model.load_state_dict(state_dict, strict=True)

    # Move model to the selected device and set to evaluation mode.
    model = model.to(device)
    model.eval()

    # Optionally compile the model for potential speed improvements.
    if args.compile:
        model = torch.compile(model)

    # Create the output directory if it does not exist.
    os.makedirs(args.out_path, exist_ok=True)

    # Process input: if it's a directory, process all image files; if it's a file, process that image.
    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_image_path = osp.join(args.in_path, file)
                file_base = osp.splitext(file)
                output_image_path = os.path.join(
                    args.out_path,
                    f"{file_base}.png",
                )
                process_image(input_image_path, output_image_path, model, device)
    elif args.in_path.lower().endswith((".png", ".jpg", ".jpeg")):
        file_base = osp.splitext(osp.basename(args.in_path))
        input_image_path = args.in_path
        output_image_path = os.path.join(
            args.out_path,
            f"{file_base}.png",
        )
        process_image(input_image_path, output_image_path, model, device)
    else:
        print(f"Invalid input path: {args.in_path}")


if __name__ == "__main__":
    main()
