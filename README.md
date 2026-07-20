# Intel OpenVINO Model Conversion Collection

This project's purpose is to consolidate machine-vision models that can be converted
to run on Intel platforms -- each entry takes a public model repo through the full
pipeline (source + weights, OpenVINO IR conversion, `benchmark_app` numbers on Intel
CPU/iGPU, numerical validation, and a runnable demo) so the result is a reproducible,
deployable IR rather than just a conversion script.

- **Platform:** Windows 10 IoT Enterprise LTSC 2021
- **OpenVINO runtime:** 2026.1.0-21367
- **Hardware:** Intel(R) Core(TM) Ultra 7 265H (4p8e2lpe + 8 Xe GPU)
- **Precision:** FP16 IR, validated against same-precision (CPU FP16) baseline

## 1. Currently supported

| Model | Directory |
| --- | --- |
| PointNet -- point-cloud semantic segmentation (S3DIS, 13 classes) | [export_pointnet/](export_pointnet/) |

The table and per-model writeups in this README also cover models converted in
earlier work on this project; only the directories listed above are present in
this checkout. See each directory's own README for full reproduction steps.

---

## 2. Directory Layout Convention

Each model directory follows this pattern:

```
export_<model>/
├── converter/              # *.xml/*.bin IR + convert script
├── benchmark/              # benchmark_cpu_result.txt, benchmark_gpu_result.txt, benchmark_app_usage.md
├── demo/                   # infer_demo.py + sample input/output artifacts
├── validation/             # validate.py, validation_report.md, metrics.json, diff_visualization.png
├── <original_repo>/        # upstream source tree
└── *_conversion_report.md  # per-model conversion writeup
```
