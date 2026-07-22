import argparse
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


ROOT = Path(__file__).resolve().parents[1]


def load_inputs(example_dir: Path):
    image_bgr = cv2.imread(str(example_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(example_dir / "rgb.png")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image_rgb, (2, 0, 1))[None]

    depth_mm = cv2.imread(str(example_dir / "raw_depth.png"), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(example_dir / "raw_depth.png")
    depth = depth_mm.astype(np.float32)[None] / 1000.0

    intrinsics = np.loadtxt(example_dir / "intrinsics.txt").astype(np.float32)
    height, width = depth.shape[-2:]
    intrinsics[0] /= width
    intrinsics[1] /= height
    return image, depth, intrinsics[None]


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    batch, height, width = depth.shape
    fx = intrinsics[:, 0, 0] * width
    fy = intrinsics[:, 1, 1] * height
    cx = intrinsics[:, 0, 2] * width
    cy = intrinsics[:, 1, 2] * height
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    xx = np.broadcast_to(xx, (batch, height, width))
    yy = np.broadcast_to(yy, (batch, height, width))
    z = depth
    x = (xx - cx[:, None, None]) * z / fx[:, None, None]
    y = (yy - cy[:, None, None]) * z / fy[:, None, None]
    return np.stack([x, y, z], axis=-1)


def save_depth_png(depth: np.ndarray, path: Path):
    finite = np.isfinite(depth) & (depth > 0)
    values = depth[finite]
    if values.size == 0:
        colored = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        lo, hi = np.percentile(values, [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), colored)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LingBot-Depth OpenVINO inference on an example scene.")
    parser.add_argument("--model", default=str(ROOT / "converter" / "lingbot_depth.xml"))
    parser.add_argument("--example-dir", default=str(ROOT / "lingbot-depth" / "examples" / "0"))
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--output-dir", default=str(ROOT / "demo" / "result"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image, depth, intrinsics = load_inputs(Path(args.example_dir))
    core = ov.Core()
    compiled = core.compile_model(args.model, args.device)
    result = compiled({"image": image, "depth": depth})
    outputs = list(result.values())
    refined_depth = np.asarray(outputs[0], dtype=np.float32)
    mask = np.asarray(outputs[1]) > 0.5 if len(outputs) > 1 else np.ones_like(refined_depth, dtype=bool)
    refined_depth = np.where(mask, refined_depth, np.inf)
    points = depth_to_points(refined_depth, intrinsics)

    np.save(output_dir / "depth_refined.npy", refined_depth.squeeze(0))
    np.save(output_dir / "points.npy", points.squeeze(0))
    save_depth_png(refined_depth.squeeze(0), output_dir / "depth_refined.png")
    print(f"Saved depth and point outputs to {output_dir}")
    print(f"Depth output shape: {refined_depth.shape}, point output shape: {points.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())