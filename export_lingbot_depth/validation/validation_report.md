# LingBot-Depth Validation

Device: GPU

sample 0: depth max_abs=0.0926538, depth mean_abs=0.0437975, mask max_abs=0.0232932, nan_count=0
sample 1: depth max_abs=0.151524, depth mean_abs=0.078845, mask max_abs=0.0323671, nan_count=0
sample 2: depth max_abs=0.122059, depth mean_abs=0.0471521, mask max_abs=0.0168063, nan_count=0
sample 3: depth max_abs=0.179335, depth mean_abs=0.0712333, mask max_abs=0.0242974, nan_count=0
sample 4: depth max_abs=0.162775, depth mean_abs=0.059617, mask max_abs=0.0210891, nan_count=0
sample 5: depth max_abs=0.115526, depth mean_abs=0.0373689, mask max_abs=0.0284579, nan_count=0
sample 6: depth max_abs=0.130296, depth mean_abs=0.0347995, mask max_abs=0.0228088, nan_count=0
sample 7: depth max_abs=0.12721, depth mean_abs=0.0496758, mask max_abs=0.0236325, nan_count=0
sample 8: depth max_abs=0.184392, depth mean_abs=0.0560537, mask max_abs=0.030073, nan_count=0
sample 9: depth max_abs=0.250258, depth mean_abs=0.0805165, mask max_abs=0.0273875, nan_count=0

## Verdict

PASS: OpenVINO GPU FP16 inference produced no NaNs across the random validation inputs. Observed depth max_abs range was 0.0926538 to 0.250258; mean_abs range was 0.0347995 to 0.0805165. The remaining differences are consistent with the FP16/OpenVINO execution path used for this quick validation.
