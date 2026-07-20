"""One-off experiment: does dropping the unused trans_feat output (via the
PointNetSemSegOV wrapper) actually change OpenVINO performance, or is it
purely a convenience wrapper? Converts both variants and leaves two IRs on
disk for benchmark_app comparison.
"""
import os
import sys

import openvino as ov
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "PointNet", "models"))
from pointnet_sem_seg import get_model  # noqa: E402

ckpt_path = os.path.join(os.path.dirname(__file__), "..", "weights", "pointnet_sem_seg_best.pth")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

dummy_input = torch.randn(1, 9, 4096)


class PointNetSemSegOV(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.model = get_model(num_class=num_classes)

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


# Variant A: raw upstream model, tuple output (logits, trans_feat) kept.
raw_model = get_model(num_class=13)
raw_model.load_state_dict(checkpoint["model_state_dict"])
raw_model.eval()
ov_model_tuple = ov.convert_model(raw_model, example_input=dummy_input, input=[(1, 9, 4096)])
ov.save_model(ov_model_tuple, "pointnet_sem_seg_tuple_output.xml", compress_to_fp16=True)
print(f"[tuple output] num_outputs = {len(ov_model_tuple.outputs)}")

# Variant B: wrapper, single output (logits only) -- what convert.py actually ships.
wrapped_model = PointNetSemSegOV(num_classes=13)
wrapped_model.model.load_state_dict(checkpoint["model_state_dict"])
wrapped_model.eval()
ov_model_single = ov.convert_model(wrapped_model, example_input=dummy_input, input=[(1, 9, 4096)])
ov.save_model(ov_model_single, "pointnet_sem_seg_single_output.xml", compress_to_fp16=True)
print(f"[single output] num_outputs = {len(ov_model_single.outputs)}")
