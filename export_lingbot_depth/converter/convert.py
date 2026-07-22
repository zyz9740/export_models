import argparse
import os
import sys
from pathlib import Path

import openvino as ov
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "lingbot-depth"
DEFAULT_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


class LingBotDepthExportWrapper(torch.nn.Module):
    # Export-only wrapper: freezes num_tokens as a constant (upstream forward
    # takes it as a traced argument), forces a static-shaped tensor graph via
    # enable_depth_mask=False, and flattens the dict output into a fixed
    # (depth, mask) tensor tuple so ov.convert_model sees a stable signature.
    def __init__(self, model: torch.nn.Module, num_tokens: int):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)

    def forward(self, image: torch.Tensor, depth: torch.Tensor):
        output = self.model.forward(
            image=image,
            num_tokens=self.num_tokens,
            depth=depth,
            enable_depth_mask=False,
        )
        depth_reg = output["depth_reg"]
        mask = output.get("mask")
        if mask is None:
            # mask_head is optional in some configs; synthesize an all-valid
            # mask so the traced graph always returns two outputs, not one.
            mask = torch.ones_like(depth_reg, dtype=torch.bool)
        # Match dtype across both branches (real sigmoid mask vs. bool
        # fallback) so the single return statement traces one fixed dtype.
        return depth_reg, mask.to(dtype=depth_reg.dtype)


def patch_nan_to_num_for_export():
    # ov.convert_model fails tracing aten::nan_to_num directly. Monkey-patch
    # torch.nan_to_num process-wide (export script only, upstream source is
    # untouched) with an equivalent where(isfinite(x), x, nan) that OpenVINO
    # can trace: isfinite + where + full_like are all natively supported.
    def export_friendly_nan_to_num(input_tensor, nan=0.0, posinf=None, neginf=None, out=None):
        if out is not None:
            raise NotImplementedError("out= is not supported by the export patch")
        replacement = torch.full_like(input_tensor, float(nan))
        return torch.where(torch.isfinite(input_tensor), input_tensor, replacement)

    torch.nan_to_num = export_friendly_nan_to_num


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LingBot-Depth to OpenVINO IR FP16.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local model.pt path.")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR), help="Path to cloned lingbot-depth source.")
    parser.add_argument("--output-dir", default=str(ROOT / "converter"), help="Directory for IR output.")
    parser.add_argument("--height", type=int, default=480, help="Static input height.")
    parser.add_argument("--width", type=int, default=640, help="Static input width.")
    parser.add_argument("--num-tokens", type=int, default=1200, help="Static token budget used by the model.")
    parser.add_argument("--name", default="lingbot_depth", help="Output model base name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(source_dir))
    from mdm.model.v2 import MDMModel

    # xformers' memory-efficient attention kernels aren't traceable by
    # ov.convert_model; disabling forces the backbone to fall back to
    # torch's native scaled_dot_product_attention, which OpenVINO maps
    # directly to an optimized SDPA kernel.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    torch.set_grad_enabled(False)
    patch_nan_to_num_for_export()

    model_id_or_path = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    print(f"Loading LingBot-Depth model: {model_id_or_path}")
    model = MDMModel.from_pretrained(model_id_or_path).eval().cpu()
    # Upstream-provided export switch: turns off antialiased interpolate
    # (antialias=True is not export-friendly) without touching source code.
    model.encoder.onnx_compatible_mode = True

    wrapped = LingBotDepthExportWrapper(model, num_tokens=args.num_tokens).eval()
    # Dummy inputs only need matching shape/dtype for tracing; values are
    # irrelevant since ov.convert_model traces the static computation graph.
    image = torch.zeros((1, 3, args.height, args.width), dtype=torch.float32)
    depth = torch.ones((1, args.height, args.width), dtype=torch.float32)

    print("Converting with openvino.convert_model...")
    ov_model = ov.convert_model(
        wrapped,
        example_input=(image, depth),
        input=[
            ("image", [1, 3, args.height, args.width], ov.Type.f32),
            ("depth", [1, args.height, args.width], ov.Type.f32),
        ],
    )

    ov_model.outputs[0].get_tensor().set_names({"depth"})
    ov_model.outputs[1].get_tensor().set_names({"mask"})
    output_path = output_dir / f"{args.name}.xml"
    ov.save_model(ov_model, output_path, compress_to_fp16=True)
    print(f"Saved OpenVINO IR: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())