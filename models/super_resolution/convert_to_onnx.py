import torch
from basicsr.archs.atd_arch import ATD
import argparse

model_path = {
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

parser = argparse.ArgumentParser()
parser.add_argument("--use_cpu", action="store_true", help="Force to use CPU")
args = parser.parse_args()


def main():
    device = "cuda" if (torch.cuda.is_available() and not args.use_cpu) else "cpu"

    model = ATD(
        upscale=2,
        embed_dim=210,
        depths=[
            6,
            6,
            6,
            6,
            6,
            6,
        ],
        num_heads=[
            6,
            6,
            6,
            6,
            6,
            6,
        ],
        window_size=16,
        category_size=256,
        num_tokens=128,
        reducted_dim=20,
        convffn_kernel_size=5,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        use_checkpoint=False,
    )

    state_dict = torch.load(model_path["classical"][str(2)], map_location=device)[
        "params_ema"
    ]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256).to(device)  # adjust size accordingly
    torch.onnx.export(model, dummy_input, "model.onnx")


if __name__ == "__main__":
    main()
