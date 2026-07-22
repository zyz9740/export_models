# LingBot-Depth OpenVINO Export

This directory contains a reproducible OpenVINO conversion scaffold for `Robbyant/lingbot-depth`.

Status: conversion completed successfully. The generated OpenVINO IR is in `converter/lingbot_depth.xml` and `converter/lingbot_depth.bin`.

## Source

The source repository was cloned into `lingbot-depth/` from:

```text
https://github.com/Robbyant/lingbot-depth
```

## Weights

Recommended model:

```text
robbyant/lingbot-depth-pretrain-vitl-14-v0.5
```

Expected checkpoint file name used by the project is `model.pt`.

Download it from one of these pages and place it locally, for example at `export_lingbot_depth/weights/model.pt`:

```text
https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/tree/main
https://www.modelscope.cn/models/Robbyant/lingbot-depth-pretrain-vitl-14-v0.5
```

The downloaded checkpoint is stored at `weights/model.pt`.

## Convert

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\converter\convert.py --model export_lingbot_depth\weights\model.pt
```

The converter emits:

```text
export_lingbot_depth/converter/lingbot_depth.xml
export_lingbot_depth/converter/lingbot_depth.bin
```

Default static shape is `image [1,3,480,640]` and `depth [1,480,640]`, matching example scene `0`. Override with `--height`, `--width`, and `--num-tokens` if needed. The export wrapper uses `enable_depth_mask=False` so the model has a static tensor graph suitable for OpenVINO conversion.

## Benchmark

```powershell
benchmark_app -m export_lingbot_depth\converter\lingbot_depth.xml -d GPU -hint latency -infer_precision f16 | Tee-Object -FilePath export_lingbot_depth\benchmark\benchmark_gpu_result.txt
```

Measured GPU result on this machine: median latency `202.51 ms`, throughput `4.89 FPS`.

## Validate

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\validation\validate.py --model export_lingbot_depth\weights\model.pt
```

## Demo

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\demo\infer_demo.py --device GPU
```

The demo writes `depth_refined.npy`, `points.npy`, and `depth_refined.png` under `export_lingbot_depth/demo/result/`.