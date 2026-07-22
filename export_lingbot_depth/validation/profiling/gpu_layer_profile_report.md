# LingBot-Depth OpenVINO IR -- Per-Layer GPU Profiling

Scope: GPU only, per user instruction (no CPU conversion, no CPU profiling in this pass).

## Method

Per-layer timing captured with `benchmark_app -exec_graph_path`, following
`openvino-converter`'s per-layer profiling method (lighter-weight than a full
VTune `gpu-hotspots` pass; sufficient for "which layer is slow and what kernel
did it get").

Command:

```powershell
benchmark_app -m export_lingbot_depth\converter\lingbot_depth.xml -d GPU -hint latency -infer_precision f16 `
    -niter 50 -exec_graph_path export_lingbot_depth\validation\profiling\exec_graph_gpu.xml
```

Artifacts:
- `exec_graph_gpu.xml` -- full per-layer runtime graph (this is the data source for everything below)
- `exec_graph_gpu.bin` -- empty placeholder, not real weights, ignore
- `exec_graph_gpu_run.log` -- full benchmark_app console log for this run
- `parse_exec_graph.py` -- reusable parser used to produce the tables below (`python parse_exec_graph.py exec_graph_gpu.xml --top 30`)

### Caveat on absolute numbers

This run's own headline latency (median **1040.13 ms**, from `exec_graph_gpu_run.log`)
is ~5x the uninstrumented baseline (median **202.51 ms**, from
`benchmark/benchmark_gpu_result.txt`, same IR, same `-hint latency -infer_precision f16`
contract, no `-exec_graph_path`). This inflation is expected -- `-exec_graph_path`
adds per-layer instrumentation overhead -- and matches the skill's documented
"absolute numbers should be cross-checked against the uninstrumented benchmark
log" caveat. **Use this capture only for relative proportions (% of total, which
layer/op-type dominates), not as a replacement for the 202.51 ms baseline
figure.**

The sum of all captured `execTimeMcs` is 946.70 ms, consistent with the
1040.13 ms instrumented median (the ~93 ms gap is host dispatch/sync overhead
not attributed to any single layer).

## Headline numbers

| Metric | Value | Source |
|---|---|---|
| Baseline GPU median latency (uninstrumented) | 202.51 ms | `benchmark/benchmark_gpu_result.txt` |
| Baseline GPU throughput | 4.89 FPS | `benchmark/benchmark_gpu_result.txt` |
| Instrumented run median latency | 1040.13 ms | `exec_graph_gpu_run.log` |
| Sum of per-layer execTimeMcs | 946.70 ms | `exec_graph_gpu.xml` (this capture) |
| Total IR layers | 671 | `exec_graph_gpu.xml` |
| Layers actually executed (rest folded/constant/dead) | 433 | `exec_graph_gpu.xml` |

All percentages below are relative to the 946.70 ms instrumented total, which
is the only self-consistent denominator available from this capture.

## Top 15 hottest individual layers

| execTimeMcs (us) | % of total | Type | Primitive (kernel) | Layer |
|---|---|---|---|---|
| 20060 | 2.12% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.3.attn` |
| 18563 | 1.96% | Pad | `border_gpu_ref__f16` | `neck.resamplers.3.1` |
| 18191 | 1.92% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.16.attn` |
| 17240 | 1.82% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.22.attn` |
| 16957 | 1.79% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.10.attn` |
| 16697 | 1.76% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.4.attn` |
| 15578 | 1.65% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.2.attn` |
| 15492 | 1.64% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.23.attn` |
| 15388 | 1.63% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.9.attn` |
| 14823 | 1.57% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.21.attn` |
| 14521 | 1.53% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.17.attn` |
| 14057 | 1.48% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.14.attn` |
| 12712 | 1.34% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.15.attn` |
| 12617 | 1.33% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.7.attn` |
| 12353 | 1.30% | scaled_dot_product_attention | `ocl::sdpa::opt__f16` | `encoder.backbone.blocks.11.attn` |

Full top-30 available by re-running `parse_exec_graph.py`.

## Aggregate by op type

| Total time | % of total | Layer count | Op type |
|---|---|---|---|
| 339.5 ms | 35.86% | 96 | **FullyConnected** |
| 303.8 ms | 32.09% | 24 | **scaled_dot_product_attention** |
| 79.0 ms | 8.34% | 36 | **Pad** |
| 77.1 ms | 8.14% | 52 | Convolution |
| 43.8 ms | 4.63% | 49 | Permute |
| 37.6 ms | 3.98% | 49 | MVN |
| 31.2 ms | 3.30% | 80 | Reorder |
| 13.0 ms | 1.37% | 6 | Resample |
| 12.5 ms | 1.33% | 9 | Deconvolution |
| 5.2 ms | 0.55% | 13 | Activation |
| < 2 ms each | < 0.2% | -- | select, Eltwise, Concat, Result, Gather, Input |

**FullyConnected + scaled_dot_product_attention together account for ~68% of
GPU time** -- this is a ViT-style transformer backbone (`encoder.backbone.blocks.0`
through at least `.23`, i.e. a 24-block encoder), and its linear projections
(qkv, mlp.fc1/fc2) plus attention are, unsurprisingly, where the bulk of
compute goes. This is expected/healthy for a transformer and not itself an
actionable inefficiency -- it is the model doing its job.

## Reference/fallback kernel finding (actionable)

Filtering by `primitiveType` containing `_ref` (OpenVINO's naming convention
for unoptimized/reference-path GPU kernels, as opposed to tuned kernels like
`jit:gemm:any`, `ocl::sdpa::opt`, `mvn_gpu_bfyx_opt`):

| Total ref-kernel time | % of total | Layer count |
|---|---|---|
| **124.4 ms** | **13.1%** | 84 layers (all `Pad` and `Permute`) |

Breakdown:
- **`Pad` -> `border_gpu_ref__f16`**: all 36 executed `Pad` layers (79.0 ms, 8.3% of total) run on the reference border-padding kernel. There is no tuned/opt variant selected for any of them. The three costliest are the final-stage resamplers in `neck`, `depth_head`, and `mask_head` (`resamplers.3.1`, 18.6 / 12.2 / 7.3 ms respectively) -- these operate on the largest spatial resolution in each head's upsampling path, which is why they dominate the Pad total.
- **`Permute` -> `permute_ref__f16`**: a subset of the 49 `Permute` layers (attention head reshape/transpose ops inside `encoder.backbone.blocks.*.attn`) also land on the reference kernel rather than `permute_f_y_axes__f16` (which appears once, at far lower cost, elsewhere in the graph). These are individually smaller (1.6-4.5 ms each) but numerous.

This is the clearest optimization signal from this profile: **13% of GPU
inference time is spent on reference-path kernels for two op types
(`Pad`, `Permute`) that have zero algorithmic reason to be slow** -- padding
and transposition are memory-layout operations, not compute-bound math. A
tuned/vectorized kernel selection (or restructuring the graph to avoid the
`Pad` altogether, e.g. by pre-padding feature maps earlier in a fused op, or
choosing a memory layout that avoids the explicit `Permute`) is the most
promising lead for a follow-up optimization pass, well ahead of touching the
attention/FullyConnected layers that already run on optimized kernels.

## Where to look next

1. **`Pad` reference-kernel usage (79 ms, 8.3%)** -- investigate why OpenVINO's
   GPU plugin didn't select an optimized border-pad kernel for this IR. Check
   whether the padding mode/parameters (`aten::pad` mode, non-power-of-2 sizes,
   asymmetric padding) are forcing the fallback, and whether reordering the
   pad relative to the neighboring conv/resample could let it fuse away
   entirely.
2. **`Permute` reference-kernel usage inside attention blocks** -- see if qkv
   reshape/transpose can be restructured (e.g. different layout choice on the
   preceding `FullyConnected`) to hit `permute_f_y_axes` or avoid the permute.
3. This is squarely the "5-15% of total latency, worth investigating" band the
   skill defines for deciding whether a fused custom-op port is worthwhile
   (Section 7.3) -- **but per the CUDA fused-op migration trigger conditions,
   proceeding down that path requires a hand-written CUDA kernel for the fused
   op in the source repo.** No such kernel was found for `Pad`/`Permute` in
   `lingbot-depth` during the original conversion; this would need graph-level
   IR optimization (kernel selection / graph rewrite) rather than a custom-op
   port. Flagging as a finding, not opening Stage 8.

## Files

```
validation/profiling/
  exec_graph_gpu.xml               committed -- the captured per-layer runtime graph
  exec_graph_gpu.bin                (derived placeholder, empty, ignore)
  exec_graph_gpu_run.log           committed -- full benchmark_app log for this run
  parse_exec_graph.py              committed -- reusable parser/reporter
  gpu_layer_profile_report.md      this file
```
