# Intel OpenVINO Model Conversion Collection

A catalog of 10 deep-learning models converted to Intel OpenVINO IR (FP16) and benchmarked on Intel CPU + iGPU.

- **Platform:** Windows 10 IoT Enterprise LTSC 2021
- **OpenVINO runtime:** 2026.1.0-21367 (most), 2026.0.0-20965 (PointNet CPU)
- **Hardware:** Intel CPU + Intel integrated GPU
- **Precision:** FP16 IR, validated against same-precision (CPU FP16) baseline

---

## 1. Summary Table

| # | Model | Task | Status | CPU FPS | GPU FPS | GPU ms | Demo | Validation |
|---|---|---|---|---:|---:|---:|:---:|:---:|
| 1 | [export_KPConv](export_KPConv/) | Point-cloud segmentation (S3DIS, 13 cls) | OK | 29.89 | **111.86** | 4.22 | .pcd x2 | PASS |
| 2 | [export_pointnet](export_pointnet/) | Point-cloud segmentation (S3DIS, 13 cls) | OK | 75.35 | 558.91 | 1.76 | txt | PASS |
| 3 | [dgcnn](dgcnn/) | Point-cloud classification (ModelNet40) | OK | 130.55 | 26.01 | 151.46 | txt | — |
| 4 | [export_dinov2](export_dinov2/) | ViT-S/14 image embedding backbone | OK | 5.94 | 25.64 | 153.63 | 10 files | PASS |
| 5 | [export_dncnn](export_dncnn/) | Grayscale denoising (sigma=25) | OK | 11.71 | **116.97** | 4.25 | 4-panel | PASS |
| 6 | [export_mimo_unet_shdocs](export_mimo_unet_shdocs/) | Specular highlight removal (MIMO-UNet) | OK | 2.27 | 22.93 (256) / **134.93** (128 base) | 43.6 / 7.41 | 3 imgs | PASS |
| 7 | [export_restormer](export_restormer/) | Defocus deblur (transformer) | OK | 0.50 | 1.02 | 980.37 | side-by-side | PASS |
| 8 | [export_specularitynet_psd](export_specularitynet_psd/) | Specular highlight removal | OK* | 1.45 | 10.03 | 99.55 | 3 imgs | PASS |
| 9 | [export_tshrnet](export_tshrnet/) | Specular highlight removal (4-UNet cascade) | OK | 11.48 | 33.44 | 30.48 | 6 imgs | PASS x4 |
| 10 | [export_zero_dce](export_zero_dce/) | Low-light enhancement | OK | 13.09 | 32.45 | 30.29 | before/after | PASS |

> \* `specularitynet_psd`: conversion and graph are correct, but upstream never released pretrained weights — runs with kaiming-init random weights.
> **Bold GPU FPS** = meets 100 FPS deployment target (4 of 10).
> Latency = median latency-hint value at typical input size; see each model's `benchmark/` folder for full numbers.

---

## 2. Status Breakdown

**10 / 10 SUCCESS:** KPConv, PointNet, dgcnn, dinov2, dncnn, mimo_unet_shdocs, restormer, specularitynet_psd, tshrnet, zero_dce.

Two earlier entries (`export_dhan_shr`, `export_icassp2024`) were removed from this catalog: DHAN-SHR could not be converted (complex-tensor FFT ops are not expressible in ONNX), and the ICASSP 2024 highlight-removal model has an upstream bug (`ConvTranspose2d` re-initialised every `forward()`) that makes the pretrained checkpoint non-reproducible. Neither was recommended for deployment.

---

## 3. Per-Model Details

### 1. export_KPConv — Point-cloud semantic segmentation (S3DIS)
- **IR:** `export_KPConv/converter/kpconv_s3dis_light_simplified.{xml,bin}` (Path B, default) and `kpconv_s3dis_light_direct.{xml,bin}` (Path A), Light_KPFCNN variant
- **Benchmark** (FP16 IR, `-hint latency`, 100 iter, valid index tensors):
  - ONNX path: CPU 29.89 FPS / 33.39 ms median · GPU **111.86 FPS / 4.22 ms median**
  - Direct path: CPU 28.74 FPS / 34.38 ms median · GPU 107.87 FPS / 4.41 ms median
- **Input caveat:** random-fill breaks gather ops; benchmark must use pre-generated valid index tensors under `benchmark/inputs/` and `benchmark/inputs_direct/`
- **Demo:** `demo/infer_demo.py` + `real_cat.pcd`, `real_table_scene_lms400.pcd`, `sample_result.txt`
- **Validation:** PyTorch CPU FP16 (autocast) vs OV GPU FP16 — mean \|Δ\| = 2.1e-3, max \|Δ\| = 1.3e-2, top-1 512/512 = 100% → **PASS** (relaxed FP16 thresholds)

### 2. PointNet — Point-cloud semantic segmentation (S3DIS)
- **IR:** `export_pointnet/converter/pointnet_sem_seg_simplified.{xml,bin}`, **static** input `[1,9,4096]`
- **Benchmark:** CPU 75.35 FPS / 12.97 ms median · GPU **558.91 FPS / 1.76 ms** median (FP16, latency hint; CPU run without the FP16 precision hint — unsupported on this CPU plugin build, see `export_pointnet/benchmark/benchmark_app_usage.md`)
- **Demo:** `demo/infer_demo.py` + `sample_result.txt`
- **Validation:** OV-GPU-FP16 vs Torch-CPU-FP16/FP32, 13 inputs → **PASS / PASS** (see `export_pointnet/validation/validation_report.md`)
- **Correctness note:** an earlier conversion used a fully dynamic input shape and reported ~2201 FPS on GPU — that IR was later found, via validation, to return numerically wrong output on GPU specifically (top-1 match as low as 0.34% vs the CPU reference on some inputs), despite running at that speed. Forcing a static `[1,9,4096]` input shape fixed it; the corrected IR is what's benchmarked and validated above. Fast-but-wrong is why validation is required, not optional.

### 3. dgcnn — Point-cloud classification (ModelNet40)
- **IR:** `dgcnn/converter/dgcnn_simplified.{xml,bin}`, input `[B, 3, 1024]`
- **Benchmark:** CPU 130.55 FPS / 37.67 ms · GPU 26.01 FPS / 151.46 ms (100 iter) — GPU slower than CPU, likely kNN / gather-heavy graph
- **Demo:** `demo/infer_demo.py` + `sample_result.txt`
- **Validation:** not performed

### 4. export_dinov2 — DINOv2 ViT-S/14 image backbone
- **IR:** FP16 via PT→ONNX→simplify→OV, input `1x3x518x518`, output CLS embedding `[1, 384]`
- **Benchmark:** CPU 5.94 FPS / 662.75 ms · GPU 25.64 FPS / 153.63 ms (100 iter, throughput hint)
- **Demo:** 10 files — `infer_demo.py` (single-image embed), `infer_demo2.py` (pairwise cosine similarity), sample images (`cat.jpg`, `dog2.jpg`, `dog3.jpg`, `sample.jpg`, `sample_variant.jpg`), `inference_output.npy`, `inference_result.png`, `similarity_result.png`
- **Validation:** CPU FP16 vs GPU FP16 — mean |Δ| = 5.81e-3, max |Δ| = 2.19e-2, cosine = 0.999996 → **PASS**

### 5. export_dncnn — DnCNN grayscale denoiser (sigma=25)
- **Conversion path:** direct PT→OV (Path A)
- **Benchmark:** CPU 11.71 FPS / 89.72 ms · GPU **116.97 FPS / 4.25 ms** median (FP16, latency hint, 256x256) — **~21x GPU speedup**
- **Demo:** `denoise_demo.png` — 4-panel comparison on Set12/05.png; PSNR 20.25 → 30.37 dB (+10.12 dB)
- **Validation:** CPU FP16 vs GPU FP16 — mean |Δ| = 9e-5, max |Δ| = 9.77e-4 → **PASS** (`diff_visualization.png`)

### 6. export_mimo_unet_shdocs — MIMO-UNet(+) specular highlight removal
- **Variants (4 IRs):** Plus+SHDocs @ 256 / 128; base+GoPro @ 256 / 128
- **Benchmark key numbers (FP16, latency hint):**
  - Plus @ 256: CPU 2.27 FPS / GPU 22.93 FPS
  - Plus @ 128: GPU 66.39 FPS
  - base @ 256: GPU 41.91 FPS
  - base @ 128: GPU **134.93 FPS** (only variant meeting 100 FPS)
- **Demo:** `infer_demo.py`, `sample_input.png`, `sample_output.png`, `sample_residual.png`
- **Validation:** @ 1x3x256x256 — mean 8.70e-5, max 9.77e-4 → **PASS**

### 7. export_restormer — Defocus deblur transformer (CVPR 2022)
- **IR:** direct OV from author TorchScript, dynamic H/W, ~26 M params
- **Benchmark:** CPU 0.50 FPS / 1979.71 ms · GPU 1.02 FPS / 980.37 ms (FP16, 256x256) — ~2x GPU, low absolute throughput
- **Demo:** `couple_defocused.jpg` → `couple_restored.png` + `demo_result.png` side-by-side; Laplacian sharpness 343.98 → 371.14 (+8 %)
- **Validation:** TorchScript CPU FP16 vs OV GPU FP16 — mean 1.79e-4, max 1.47e-3 → **PASS**

### 8. export_specularitynet_psd — SpecularityNet-PSD (random weights)
- **Status caveat:** upstream never released pretrained weights (GitHub #7 open since 2022); convert.py uses kaiming-init random weights. Graph is correct, outputs arbitrary.
- **Benchmark:** CPU 1.45 FPS / 691.75 ms · GPU 10.03 FPS / 99.55 ms (FP16, latency hint)
- **Demo:** `infer_demo.py`, `sample_input.png`, `sample_output.png` (noise due to random weights), `sample_residual.png`
- **Validation:** PT CPU FP16 vs OV GPU FP16 (same random seed) — mean 3.94e-4, max 7.81e-3 → **PASS**

### 9. export_tshrnet — TSHRNet 3-stage highlight removal (ICCV 2023)
- **Architecture:** 3-stage cascade of 4 UNets, 116.99 M params, 1x3x256x256. 100-FPS target not met (intrinsic to architecture).
- **Benchmark:** CPU 11.48 FPS / 92.22 ms · GPU.0 33.44 FPS / 30.48 ms (FP16, latency hint, iGPU)
- **Demo:** 6 images — `sample_input.png`, `sample_output.png`, `sample_albedo.png`, `sample_shading.png`, `sample_refined.png`, `sample_residual.png` (full cascade visualised)
- **Validation:** 4 outputs individually — albedo 2.23e-4/1.71e-3, shading 2.81e-4/2.38e-3, diffuse_refined 1.81e-4/2.56e-3, diffuse_tc 1.71e-4/2.44e-3 → all **PASS**

### 10. export_zero_dce — Zero-DCE low-light enhancement
- **Conversion path:** direct PT→OV (Path A), dynamic H/W
- **Benchmark:** CPU 13.09 FPS / 77.92 ms · GPU 32.45 FPS / 30.29 ms median (FP16, latency hint, 512x512) — ~2.6x GPU speedup
- **Demo:** `infer_demo.py`, `lowlight_sample.jpg` → `enhanced_sample.jpg`
- **Validation:** PT CPU FP16 vs OV GPU FP16 @ 1x3x512x512 — mean 1.32e-6, max 9.77e-4 → **PASS**

---

## 4. Cross-Cutting Observations

- **Models meeting 100 FPS GPU target (4):** PointNet (559), export_KPConv (112), MIMO-UNet base @ 128 (135), DnCNN (117).
- **Largest GPU speedup vs CPU:** PointNet ~7.4x, DnCNN ~21x (CPU here excludes the FP16 precision hint, unsupported on this CPU plugin build — see export_pointnet's benchmark notes), export_KPConv ~3.7x. One model is GPU-slower (dgcnn) due to kNN/gather ops.
- **Validation coverage:** 9 models have full same-precision CPU FP16 vs GPU FP16 validation with passing metrics (7 image/vision + export_KPConv + PointNet). 1 point-cloud model (dgcnn) has no validation folder.
- **Benchmark-file convention:** `benchmark_cpu_result.txt`, `benchmark_gpu_result.txt`, and `benchmark_app_usage.md` in each `benchmark/` folder.
- **Demo-folder convention:** Each success case ships an `infer_demo.py` plus input/output image(s) or a multi-panel comparison PNG.

---

## 5. Directory Layout Convention

Each model directory follows this pattern:

```
<model>/
├── converter/              # *.xml/*.bin IR + convert script
├── benchmark/              # benchmark_cpu_result.txt, benchmark_gpu_result.txt, benchmark_app_usage.md
├── demo/                   # infer_demo.py + sample input/output artifacts
├── validation/             # validate.py, validation_report.md, metrics.json, diff_visualization.png
├── <original_repo>/        # upstream source tree
└── *_conversion_report.md  # per-model conversion writeup
```
