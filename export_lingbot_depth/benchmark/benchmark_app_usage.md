# Benchmark Result

Command used:

```powershell
benchmark_app -m export_lingbot_depth\converter\lingbot_depth.xml -d GPU -hint latency -infer_precision f16 | Tee-Object -FilePath export_lingbot_depth\benchmark\benchmark_gpu_result.txt
```

Result summary from `benchmark_gpu_result.txt`:

- Device: GPU.0
- First inference: 216.74 ms
- Median latency: 202.51 ms
- Average latency: 203.77 ms
- Throughput: 4.89 FPS