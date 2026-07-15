import os
import sys

import torch.nn as nn

# Reuse the upstream PointNet semantic segmentation definition.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "PointNet", "models"))

from pointnet_sem_seg import get_model  # noqa: E402


class PointNetSemSegOV(nn.Module):
    """Wrap PointNet so OpenVINO conversion exports only segmentation logits."""

    def __init__(self, num_classes=13):
        super().__init__()
        self.model = get_model(num_class=num_classes)

    def forward(self, x):
        logits, _ = self.model(x)
        return logits
