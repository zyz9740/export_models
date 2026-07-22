# LingBot-Depth VTune XMX Analysis

Scope: OpenVINO GPU execution of `export_lingbot_depth/converter/lingbot_depth.xml`.

## Command

```powershell
benchmark_app -m export_lingbot_depth\converter\lingbot_depth.xml -d GPU -hint latency -infer_precision f16 -niter 50
```

VTune collection type: GPU Hotspots / GPU Compute Media metrics.

Primary artifacts:

- `vtune_xmx_summary.csv`
- `vtune_xmx_hotspots_by_task.csv`
- `exec_graph_gpu.xml`
- `gpu_layer_profile_report.md`

## Platform

| Item | Value |
|---|---:|
| GPU | Intel(R) Arc(TM) 140T GPU (16GB) |
| Driver | 32.0.101.8801 |
| XVE count | 128 |
| GPU time | 55.639 s |
| VTune computing task time accounted in task CSV | 50.323 s |
| XVE array stalled/idle | 84.1% |
| Occupancy | 36.9% |

## XMX Answer

Yes, the run used XMX. VTune reported non-zero `XVE Instructions:XMX instructions` and `XVE Pipelines:XMX active(%)` in the per-computing-task report.

| Metric | Value |
|---|---:|
| Total XMX instructions | 461.83 B |
| Time-weighted XMX pipeline active | 3.23% |

Interpretation: XMX is present, but it is not highly saturated. The workload spends most GPU busy time outside active XMX execution, consistent with VTune's `XVE Array Stalled/Idle = 84.1%` and moderate occupancy.

## VTune Kernel Families

| VTune kernel family | OpenVINO operator mapping | Task time | Share of task time | Weighted XMX active | XMX instructions |
|---|---|---:|---:|---:|---:|
| `gemm_kernel` | `FullyConnected` / `jit:gemm:any__f16` | 19.256 s | 38.26% | 4.51% | 243.45 B |
| `sdpa_micro*` | `scaled_dot_product_attention` / `ocl::sdpa::opt__f16` | 15.198 s | 30.20% | 2.30% | 98.45 B |
| `gen_conv` | `Convolution` | 5.249 s | 10.43% | 2.42% | 35.83 B |
| `border_gpu_ref*` | `Pad` / `border_gpu_ref__f16` | 3.163 s | 6.28% | 1.46% | 13.27 B |
| `permute_ref*` | `Permute` / `permute_ref__f16` | 2.388 s | 4.75% | 5.06% | 34.81 B |
| reorder kernels | `Reorder` | 1.833 s | 3.64% | 3.09% | 15.36 B |
| `mvn_gpu_bfyx_opt*` | `MVN` | 1.468 s | 2.92% | 3.00% | 12.90 B |
| `resample_ref*` | `Resample` | 0.844 s | 1.68% | 0.77% | 1.84 B |

## Top Individual XMX Tasks

| VTune task | Time | XMX active | XMX instructions | Instances | Average time |
|---|---:|---:|---:|---:|---:|
| `gemm_kernel` | 3.946 s | 9.60% | 106.95 B | 1224 | 3.224 ms |
| `sdpa_micro__prefill_9410584702832551303` | 15.198 s | 2.30% | 98.45 B | 1224 | 12.417 ms |
| `gemm_kernel` | 6.731 s | 3.70% | 69.66 B | 1224 | 5.499 ms |
| `gemm_kernel` | 6.684 s | 2.80% | 52.23 B | 1224 | 5.460 ms |
| `permute_ref_9624787355049321418_0_0` | 1.681 s | 6.20% | 29.81 B | 1224 | 1.374 ms |
| `gemm_kernel` | 1.895 s | 2.80% | 14.62 B | 1224 | 1.548 ms |

## OpenVINO Layer Mapping

The OpenVINO execution graph confirms the operator-level mapping:

| OpenVINO op type | Primitive | Layer time | Share of layer time | Count |
|---|---|---:|---:|---:|
| `FullyConnected` | `jit:gemm:any__f16` | 339.5 ms | 35.86% | 96 |
| `scaled_dot_product_attention` | `ocl::sdpa::opt__f16` | 303.8 ms | 32.09% | 24 |
| `Pad` | `border_gpu_ref__f16` | 79.0 ms | 8.34% | 36 |
| `Convolution` | `jit:ir__f16` | 77.1 ms | 8.14% | 52 |
| `Permute` | `permute_ref__f16` | 43.3 ms | 4.57% | 48 |

## Conclusion

LingBot-Depth does use XMX on the Arc 140T GPU, mainly through transformer `FullyConnected` GEMM kernels and the optimized SDPA attention kernel. The dominant XMX user is `FullyConnected` (`gemm_kernel`), followed by `scaled_dot_product_attention` (`sdpa_micro*`).

However, XMX utilization is low: the time-weighted XMX pipeline active value is only 3.23%, and VTune reports 84.1% XVE stalled/idle. The result looks memory/dispatch/layout limited rather than XMX-compute saturated. The clearest non-XMX optimization signal remains the reference-path `Pad` and `Permute` kernels already called out in `gpu_layer_profile_report.md`.