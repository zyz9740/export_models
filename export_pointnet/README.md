# PointNet -- OpenVINO IR Conversion

Point-cloud semantic segmentation (S3DIS, 13 classes), converted from
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
to OpenVINO FP16 IR.

## Model Info

| Item | Value |
| --- | --- |
| Model | PointNet (semantic segmentation head) |
| Task | Point cloud semantic segmentation |
| Dataset | S3DIS (13 classes), checkpoint mIoU 43.7% |
| Input | `[1, 9, 4096]` (xyz + rgb + normalized-xyz, per point) |
| Output | `[1, 4096, 13]` (log-softmax over 13 classes) |
| Precision | FP16 |
| Conversion path | Path A -- direct `ov.convert_model()`, **static input shape** |
| Validation | **PASS / PASS** (both comparisons -- see `validation/validation_report.md`) |

## Provenance

- Upstream source: `yanx27/Pointnet_Pointnet2_pytorch`, submodule at
  `export_pointnet/PointNet/` (see root [.gitmodules](../.gitmodules)), unmodified.
- `converter/model.py` reuses the upstream `pointnet_sem_seg.get_model()`
  definition directly (via `sys.path`) and wraps it in a thin
  `nn.Module` that returns only the segmentation logits -- no changes to
  upstream source files are needed.
- The pretrained checkpoint (`best_model.pth`, mIoU 43.7%) ships inside
  upstream's own `log/sem_seg/pointnet_sem_seg/checkpoints/` directory --
  it is not hosted separately. `scripts/fetch_assets.py` copies it out and
  verifies its SHA256.

## Critical finding: static input shape is required for correct GPU inference

Converting with a fully dynamic input shape
(`ov.convert_model(model, example_input=dummy_input)`, no `input=`) produces
an IR that runs correctly on CPU but returns **numerically wrong output on
GPU** -- top-1 class match against the CPU reference dropped as low as 0.34%
on some inputs. This is a GPU-plugin-specific bug triggered by the dynamic
`[?, ?, ?]` shape, not a general FP16/precision issue (CPU on the exact same
IR was correct).

`converter/convert.py` forces a static shape instead:

```python
ov_model = ov.convert_model(model, example_input=dummy_input, input=[(1, 9, 4096)])
```

Full isolation, before/after evidence, and a second (unrelated) GPU-plugin
issue found during validation -- compiled-model reuse across back-to-back
inferences with different inputs corrupting output with NaNs -- are
documented in `validation/validation_report.md`. **Do not revert to a
dynamic-shape export without re-running validation.**

## Directory Layout

```text
export_pointnet/
├── PointNet/                     # upstream source (git submodule, yanx27/Pointnet_Pointnet2_pytorch), unmodified
├── scripts/
│   └── fetch_assets.py           # copies + SHA256-verifies best_model.pth out of the submodule
├── weights/                      # pretrained weights (NOT in git, fetched via fetch_assets.py)
│   └── pointnet_sem_seg_best.pth
├── converter/
│   ├── model.py                  # PointNetSemSegOV wrapper (returns logits only)
│   ├── convert.py                # PyTorch -> OpenVINO IR, static input shape
│   ├── pointnet_sem_seg_simplified.xml   # derived, not committed
│   └── pointnet_sem_seg_simplified.bin   # derived, not committed
├── benchmark/
│   ├── benchmark_app_usage.md
│   ├── benchmark_cpu_result.txt
│   └── benchmark_gpu_result.txt
├── validation/
│   ├── validate.py                # comparison A (OV-GPU-FP16 vs Torch-CPU-FP16) + B (vs Torch-CPU-FP32)
│   ├── validation_results.json
│   └── validation_report.md
├── demo/
│   ├── infer_demo.py             # runs on a structured synthetic room point cloud
│   └── sample_result.txt
├── requirements.txt
└── README.md
```

## Reproduce from scratch

```bash
# from the export_models root
git submodule update --init export_pointnet/PointNet

pip install -r export_pointnet/requirements.txt

# fetch the checkpoint (copies out of the submodule, verifies SHA256)
python export_pointnet/scripts/fetch_assets.py

# convert to OpenVINO IR (FP16, static [1,9,4096] input)
cd export_pointnet/converter && python convert.py && cd ../..

# benchmark
cd export_pointnet/benchmark
benchmark_app -m ../converter/pointnet_sem_seg_simplified.xml -d CPU -hint latency -infer_precision f16
benchmark_app -m ../converter/pointnet_sem_seg_simplified.xml -d GPU -hint latency -infer_precision f16
cd ../..

# validate (writes validation_results.json)
cd export_pointnet/validation && python validate.py && cd ../..

# run the demo
cd export_pointnet/demo && python infer_demo.py
```

## Quick Start (already-converted IR)

```bash
pip install -r requirements.txt
cd demo && python infer_demo.py
```

## Benchmark

`benchmark_app`, FP16, `-hint latency -infer_precision f16` (GPU) /
`-hint latency` (CPU -- this build's CPU plugin does not support the
explicit FP16 inference-precision hint, see `benchmark/benchmark_app_usage.md`).
Measured against the corrected static-shape IR:

| Device | Median latency | Throughput |
| --- | --- | --- |
| CPU | 12.97 ms | 75.35 FPS |
| GPU | 1.76 ms | 558.91 FPS |

Raw logs: `benchmark/benchmark_cpu_result.txt`, `benchmark/benchmark_gpu_result.txt`.

> An earlier, unvalidated pass reported GPU throughput of ~2201 FPS. That
> number was measured against the dynamic-shape IR, which was later found
> (via `validation/validate.py`) to return numerically wrong segmentation
> output on GPU. The 558.91 FPS above is the correct figure for the fixed,
> validated (static-shape) IR -- see `validation/validation_report.md` for
> the full root-cause writeup.
