import argparse
import os
import sys

import torch
import torch.nn as nn


class SameLengthTemporalModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pad = (model.receptive_field() - 1) // 2

    def forward(self, x):
        if self.pad > 0:
            left = x[:, :1].expand(-1, self.pad, -1, -1)
            right = x[:, -1:].expand(-1, self.pad, -1, -1)
            x = torch.cat((left, x, right), dim=1)
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="diagnostics/videopose3d_export/VideoPose3D")
    parser.add_argument("--checkpoint", default="diagnostics/videopose3d_export/pretrained_h36m_detectron_coco.bin")
    parser.add_argument("--output", default="models/videopose3d_scripted.pt")
    parser.add_argument("--architecture", default="3,3,3,3,3")
    parser.add_argument("--channels", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.repo))

    from common.model import TemporalModel

    filter_widths = [int(width) for width in args.architecture.split(",")]
    model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=filter_widths,
        causal=False,
        dropout=0.25,
        channels=args.channels,
        dense=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_pos"])
    model.eval()

    wrapper = SameLengthTemporalModel(model).eval()
    example_input = torch.zeros(1, model.receptive_field(), 17, 2)
    scripted = torch.jit.trace(wrapper, example_input)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scripted.save(args.output)
    print(f"Saved TorchScript VideoPose3D model to {args.output}")
    print(f"Receptive field: {model.receptive_field()} frames")


if __name__ == "__main__":
    main()
