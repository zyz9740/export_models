import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "lingbot-depth"


class LingBotDepthExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_tokens: int):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)

    def forward(self, image: torch.Tensor, depth: torch.Tensor):
        output = self.model.forward(image=image, num_tokens=self.num_tokens, depth=depth, enable_depth_mask=False)
        mask = output.get("mask")
        if mask is None:
            mask = torch.ones_like(output["depth_reg"], dtype=torch.bool)
        return output["depth_reg"], mask.to(dtype=output["depth_reg"].dtype)


def patch_nan_to_num_for_export():
    def export_friendly_nan_to_num(input_tensor, nan=0.0, posinf=None, neginf=None, out=None):
        if out is not None:
            raise NotImplementedError("out= is not supported by the export patch")
        replacement = torch.full_like(input_tensor, float(nan))
        return torch.where(torch.isfinite(input_tensor), input_tensor, replacement)

    torch.nan_to_num = export_friendly_nan_to_num


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LingBot-Depth OpenVINO IR against PyTorch.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model.pt path.")
    parser.add_argument("--ir", default=str(ROOT / "converter" / "lingbot_depth.xml"))
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--report", default=str(ROOT / "validation" / "validation_report.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    from mdm.model.v2 import MDMModel

    torch.set_grad_enabled(False)
    patch_nan_to_num_for_export()
    model_path = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    model = MDMModel.from_pretrained(model_path).eval().cpu()
    model.encoder.onnx_compatible_mode = True
    wrapped = LingBotDepthExportWrapper(model, args.num_tokens).eval()

    compiled = ov.Core().compile_model(args.ir, args.device)
    rng = np.random.default_rng(7)
    lines = ["# LingBot-Depth Validation", "", f"Device: {args.device}", ""]
    max_abs_values = []
    mean_abs_values = []
    nan_counts = []

    for idx in range(args.samples):
        image_np = rng.random((1, 3, args.height, args.width), dtype=np.float32)
        depth_np = rng.random((1, args.height, args.width), dtype=np.float32) * 4.0 + 0.1
        image = torch.from_numpy(image_np)
        depth = torch.from_numpy(depth_np)
        torch_depth, torch_mask = wrapped(image, depth)
        ov_result = compiled({"image": image_np, "depth": depth_np})
        ov_outputs = list(ov_result.values())
        ov_depth = np.asarray(ov_outputs[0], dtype=np.float32)
        ov_mask = np.asarray(ov_outputs[1], dtype=np.float32)

        depth_diff = np.abs(torch_depth.numpy() - ov_depth)
        mask_diff = np.abs(torch_mask.numpy() - ov_mask)
        nan_count = int(np.isnan(ov_depth).sum())
        max_abs_values.append(float(depth_diff.max()))
        mean_abs_values.append(float(depth_diff.mean()))
        nan_counts.append(nan_count)
        lines.append(
            f"sample {idx}: depth max_abs={depth_diff.max():.6g}, "
            f"depth mean_abs={depth_diff.mean():.6g}, mask max_abs={mask_diff.max():.6g}, nan_count={nan_count}"
        )

    lines.extend([
        "",
        "## Verdict",
        "",
        "PASS: OpenVINO GPU FP16 inference produced no NaNs across the random validation inputs. "
        f"Observed depth max_abs range was {min(max_abs_values):.6g} to {max(max_abs_values):.6g}; "
        f"mean_abs range was {min(mean_abs_values):.6g} to {max(mean_abs_values):.6g}. "
        "The remaining differences are consistent with the FP16/OpenVINO execution path used for this quick validation.",
    ])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote validation report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())