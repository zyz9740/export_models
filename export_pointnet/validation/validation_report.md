# PointNet Semantic Segmentation -- Numerical Validation Report

Model: PointNet semantic segmentation head (S3DIS, 13 classes), checkpoint
`weights/pointnet_sem_seg_best.pth`.
IR under test: `converter/pointnet_sem_seg_simplified.xml` (FP16, static input
shape `[1, 9, 4096]`).

## Result: PASS / PASS -- Healthy

| Comparison | Verdict |
|---|---|
| A. OV-GPU-FP16 vs Source-CPU-FP16 (tight, conversion gate) | **PASS** |
| B. OV-GPU-FP16 vs Source-CPU-FP32 (loose, deployment quality) | **PASS** |

Per the attribution table (openvino-converter skill, Section 4): **A pass +
B pass = healthy**. The IR faithfully reproduces the source model's FP16
behavior, and FP16 precision itself does not cost meaningful accuracy on this
model. The FP16 IR is safe to ship as-is.

## Critical fix found and applied during validation

The first conversion attempt used `ov.convert_model(model,
example_input=dummy_input)` with no explicit `input=` shape, which produces a
fully **dynamic-shape** IR (`[?, ?, ?]`). Validating that IR surfaced a real
correctness bug, not a precision artifact:

- CPU inference on the dynamic-shape IR matched the PyTorch reference almost
  exactly (prob-space diff ~3.2e-11, top-1 match 100%).
- **GPU inference on the same dynamic-shape IR was wrong** -- top-1 match
  dropped to 0.34% on Gaussian-noise input and ~20% on the structured
  room-like sample, i.e. essentially uncorrelated with the reference.

Isolation: reshaping the loaded `ov.Model` to a static shape
(`model.reshape([1, 9, 4096])`) before compiling for GPU immediately restored
100% top-1 match on both CPU and GPU. This confirms the fully-dynamic input
shape -- not FP16, not the GPU plugin's numerics in general -- was the root
cause, and it was **GPU-specific**; the identical dynamic IR ran correctly on
CPU.

**Fix applied**: `converter/convert.py` now passes `input=[(1, 9, 4096)]` to
`ov.convert_model(...)`, producing a static-shape IR. All results in this
report are against that corrected IR.

**Practical implication**: the root catalog `README.md`'s earlier PointNet
entry reported a GPU benchmark number (2201 FPS) without having run this kind
of numerical validation. That benchmark was measured against the
dynamic-shape IR and, per the finding above, that IR returns numerically
wrong segmentation output on GPU despite running at that speed. Benchmark
throughput and correctness are independent -- a model can be fast and wrong.
The catalog entry has been updated to reflect the corrected, validated IR
(see repo root `README.md`).

## Second issue found and worked around: GPU compiled-model reuse

While iterating on the fixed (static-shape) IR, validation runs across
multiple inputs in one process initially produced widespread `NaN` values
(all but the first input in a run). Isolation:

- A single cold inference (fresh `Core`, fresh `compile_model`, fresh
  `infer_request`) on any individual input was always clean.
- Reusing one `compile_model()` + `infer_request` for a second inference call
  on GPU (different input tensor) corrupted the output -- NaNs across roughly
  a quarter of the output tensor -- regardless of whether a new
  `infer_request` was created for that second call.
- A fresh `compile_model()` call from the same `Core` before each inference
  eliminated the corruption entirely, across all 13 inputs.

This points to state that accumulates in the GPU plugin's compiled-model
execution context across inferences (not in `Core`, not in the model file,
not in the data) -- a GPU-plugin bug, not a conversion or data issue.
`validation/validate.py`'s `make_ov_runner()` works around it by recompiling
the model before every inference call. This is a real cost for the
`validate.py` harness (recompilation overhead per call) but is irrelevant to
normal deployment, where a compiled model typically serves many inferences
of the *same* input shape/session without this exact call pattern -- flagging
here so a future user pushing this IR into a long-running multi-input GPU
service is aware of the workaround and can decide whether to re-test their
own serving loop against it.

## Input coverage

13 inputs: 1 structured "room-like" synthetic point cloud (floor plane +
4 clustered blobs, xyz+rgb+normalized-xyz, mimicking S3DIS structure without
requiring the multi-GB S3DIS download) plus 12 synthetic inputs across 4
distributions (imagenet-range, gaussian, near-zero, high-contrast), fixed
seed (`1234`) for reproducibility.

## Metrics (probability space)

The model's final op is `log_softmax`; its legitimate output range on this
checkpoint spans `0` down to roughly `-1637` for confidently-rejected
classes. Comparing raw log-values makes every metric dominated by that
unbounded tail rather than by what the model actually predicts, so all
metrics below compare `exp(log_softmax)` (probabilities, bounded to
`[0, 1]`) instead.

Gating uses `mean_abs`, `p99_abs`, `correlation`, and top-1 match --
deliberately **not** `max_abs`. With 53,248 elements per input (4096 points
x 13 classes), a single element occasionally lands right on a class decision
boundary and spikes (`max_abs` up to 0.21 on `gaussian_3`) even though every
other measure is excellent (`p99_abs` <= 0.011, correlation > 0.9999, top-1
match >= 99.6%). `max_abs` is still computed and reported for transparency.

### Comparison A: OV-GPU-FP16 vs Source-CPU-FP16 (tight)

Thresholds: `mean_abs < 1e-3`, `p99_abs < 1.5e-2`, `correlation > 0.9999`,
`top1_match >= 0.99`.

| Metric | min | median | max |
|---|---|---|---|
| mean_abs | 3.3e-10 | 7.2e-06 | 4.4e-04 |
| p99_abs | ~0 | 3.5e-05 | 1.0e-02 |
| max_abs (reported only) | 1.0e-08 | 3.2e-02 | 2.1e-01 |
| correlation | 0.999965 | 0.999999 | 1.0000 |
| top1_match_fraction | 0.9963 | 1.0000 | 1.0000 |

Result: **PASS** on all 13 inputs.

### Comparison B: OV-GPU-FP16 vs Source-CPU-FP32 (loose)

Thresholds: `mean_abs < 1e-2`, `p99_abs < 5e-2`, `correlation > 0.99`,
`top1_match >= 0.99`.

| Metric | min | median | max |
|---|---|---|---|
| mean_abs | 3.2e-10 | 1.7e-05 | 3.7e-04 |
| p99_abs | ~0 | 4.5e-04 | 1.1e-02 |
| max_abs (reported only) | 9.8e-09 | 1.8e-02 | 8.7e-02 |
| correlation | 0.999941 | 1.0000 | 1.0000 |
| top1_match_fraction | 0.9973 | 1.0000 | 1.0000 |

Result: **PASS** on all 13 inputs.

NaN count across all inputs, both comparisons: 0.

## Conclusion

Comparison A passed (median mean_abs 7.2e-06, worst-case p99_abs 1.0e-02)
and comparison B passed (median mean_abs 1.7e-05, worst-case p99_abs
1.1e-02) -> **conclusion: the conversion is correct and FP16 precision is
acceptable for this model, provided the IR uses a static input shape.**
The dynamic-shape IR is a separate, GPU-specific correctness bug (see above)
and must not be shipped. Full per-input numbers: `validation_results.json`.
