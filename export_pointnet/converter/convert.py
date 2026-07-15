import os

import openvino as ov
import torch

from model import PointNetSemSegOV

ckpt_path = os.path.join(os.path.dirname(__file__), "..", "weights", "pointnet_sem_seg_best.pth")

model = PointNetSemSegOV(num_classes=13)
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 9, 4096)

# Static input shape is required: a fully dynamic [?,?,?] IR produces
# incorrect results on GPU (validated against CPU -- see
# validation/validation_report.md, comparison A). CPU is unaffected;
# only GPU diverges on the dynamic-shape IR.
ov_model = ov.convert_model(model, example_input=dummy_input, input=[(1, 9, 4096)])
ov.save_model(ov_model, "pointnet_sem_seg_simplified.xml", compress_to_fp16=True)

loaded_model = ov.Core().read_model("pointnet_sem_seg_simplified.xml")
print(f"Saved pointnet_sem_seg_simplified.xml/.bin  "
      f"input={loaded_model.inputs[0].partial_shape}  "
      f"output={loaded_model.outputs[0].partial_shape}")
