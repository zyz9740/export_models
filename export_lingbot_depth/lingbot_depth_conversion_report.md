# LingBot-Depth OpenVINO Conversion Report

Status: success.

## Source

- Repository: `https://github.com/Robbyant/lingbot-depth`
- Source directory: `export_lingbot_depth/lingbot-depth`
- Checkpoint: `export_lingbot_depth/weights/model.pt`
- Checkpoint size: `1284837952` bytes

## Conversion

Command:

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\converter\convert.py --model export_lingbot_depth\weights\model.pt
```

Output IR:

- `export_lingbot_depth/converter/lingbot_depth.xml` (`1416644` bytes)
- `export_lingbot_depth/converter/lingbot_depth.bin` (`642320108` bytes)

The exported model uses fixed inputs matching the bundled example scene:

- `image`: `[1, 3, 480, 640]`, FP32 input to the graph
- `depth`: `[1, 480, 640]`, FP32 input to the graph

The converter exports the core model outputs `depth` and `mask`. Point cloud generation is performed as demo post-processing from the OpenVINO depth output and camera intrinsics.

## Export Notes

OpenVINO conversion initially failed on `aten::nan_to_num`. The converter applies an export-only replacement equivalent to `where(isfinite(x), x, 0)` for this model path, preserving the invalid-depth cleanup behavior without changing the cloned source repository.

## Demo

Command:

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\demo\infer_demo.py --device GPU
```

Outputs:

- `export_lingbot_depth/demo/result/depth_refined.npy`
- `export_lingbot_depth/demo/result/depth_refined.png`
- `export_lingbot_depth/demo/result/points.npy`

Observed output shapes:

- Depth: `[1, 480, 640]`
- Points: `[1, 480, 640, 3]`

## Validation

Command:

```powershell
.\.venv\Scripts\python.exe export_lingbot_depth\validation\validate.py --model export_lingbot_depth\weights\model.pt --device GPU --samples 10
```

Summary:

- No NaNs across 10 random inputs.
- Depth `max_abs` range: `0.0926538` to `0.250258`.
- Depth `mean_abs` range: `0.0347995` to `0.0805165`.
- Verdict: pass for quick FP16 OpenVINO-vs-PyTorch validation.

Full report: `export_lingbot_depth/validation/validation_report.md`.

## Benchmark

Command:

```powershell
benchmark_app -m export_lingbot_depth\converter\lingbot_depth.xml -d GPU -hint latency -infer_precision f16 | Tee-Object -FilePath export_lingbot_depth\benchmark\benchmark_gpu_result.txt
```

GPU result:

- Compile model: `7033.46 ms`
- First inference: `216.74 ms`
- Count: `294 iterations`
- Duration: `60103.64 ms`
- Median latency: `202.51 ms`
- Average latency: `203.77 ms`
- Min latency: `201.61 ms`
- Max latency: `219.22 ms`
- Throughput: `4.89 FPS`