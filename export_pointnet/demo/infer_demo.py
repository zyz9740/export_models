"""PointNet semantic segmentation -- OpenVINO inference demo.

Uses a structured synthetic room point cloud (floor plane + a few clustered
furniture-like blobs) rather than pure random noise, so the output classes
are at least plausible for the input structure. Full S3DIS is a multi-GB
dataset and is not bundled with this export -- swap in your own point cloud
(shape [1, 9, 4096]: xyz + rgb + normalized-xyz) by replacing
`make_room_like_sample()`'s output with your own array.
"""

import numpy as np
import openvino as ov

CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]
NUM_POINTS = 4096


def make_room_like_sample(seed: int = 0) -> np.ndarray:
    """Synthetic room point cloud: xyz + rgb + normalized xyz, 9 channels.

    Half the points form a floor plane; the rest cluster into 4 vertical
    blobs (stand-ins for furniture/walls), matching the input structure
    PointNet's semantic-seg head expects from a real S3DIS room scan.
    """
    rng = np.random.default_rng(seed)
    n = NUM_POINTS
    xyz = np.zeros((n, 3), dtype=np.float32)

    n_floor = n // 2
    xyz[:n_floor, 0] = rng.uniform(0, 5, n_floor)
    xyz[:n_floor, 1] = rng.uniform(0, 5, n_floor)
    xyz[:n_floor, 2] = rng.normal(0.0, 0.02, n_floor)

    n_rest = n - n_floor
    cluster_centers = rng.uniform([0, 0, 0.5], [5, 5, 2.5], size=(4, 3))
    assign = rng.integers(0, 4, n_rest)
    xyz[n_floor:] = cluster_centers[assign] + rng.normal(0, 0.15, (n_rest, 3))

    rgb = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    mins = xyz.min(axis=0)
    span = np.clip(xyz.max(axis=0) - mins, 1e-6, None)
    norm_xyz = (xyz - mins) / span

    points = np.concatenate([xyz, rgb, norm_xyz], axis=1).astype(np.float32)
    return points.T[None, ...]  # [1, 9, 4096]


print("=" * 50)
print("PointNet OpenVINO Inference Demo")
print("=" * 50)

core = ov.Core()
model = core.read_model("../converter/pointnet_sem_seg_simplified.xml")
compiled_model = core.compile_model(model, "CPU")
infer_request = compiled_model.create_infer_request()
print("[1/3] Model loaded")

input_data = make_room_like_sample()
print(f"[2/3] Input prepared: shape={input_data.shape} (synthetic room-like point cloud)")

result = infer_request.infer(inputs={0: input_data})
output = result[compiled_model.output(0)]
pred_labels = np.argmax(output, axis=-1)[0]
print("[3/3] Inference done")

print(f"Output shape: {output.shape}  (log-softmax over {len(CLASSES)} classes)")
counts = np.bincount(pred_labels, minlength=len(CLASSES))
print("\nPredicted class distribution across 4096 points:")
for i, name in enumerate(CLASSES):
    if counts[i] > 0:
        print(f"  {name:10s}: {counts[i]:5d} points ({100 * counts[i] / NUM_POINTS:.1f}%)")

print(f"\nFirst point predicted class: {CLASSES[pred_labels[0]]}")

print("=" * 50)
print("Demo finished")
