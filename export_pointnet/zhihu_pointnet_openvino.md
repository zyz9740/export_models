# 用 OpenVINO 跑 PointNet：从论文到 GPU 部署，转换比想象中简单

点云深度学习绕不开一个名字——PointNet。这篇文章从它的历史讲起，拆解一下模型结构，然后展示 OpenVINO 转换 PointNet 有多省心——一段十几行的代码就能搞定，最后附上所有相关的代码仓库链接。

## 一、PointNet 的历史

PointNet 出自斯坦福大学 Charles R. Qi、Hao Su、Kaichun Mo、Leonidas J. Guibas 等人，发表于 CVPR 2017，论文题目是 [《PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation》](https://arxiv.org/abs/1612.00593)。

在它之前，处理 3D 点云的主流思路是"曲线救国"：把点云体素化（voxelize）成规则的三维网格再套 3D 卷积，或者把点云投影成多视角图像再用 2D CNN 处理。这两种方法都要绕开点云本身"无序、稀疏"的特性，代价是几何细节丢失，或者体素网格带来的立方级内存/计算开销。

PointNet 的思路很直接：点云本质上是一个**无序集合**（set），那网络就应该对点的排列顺序不敏感——你把点打乱输入，网络的输出应该完全不变（permutation invariance）。论文证明了一个很简洁的定理：只要用一个**对称函数**（symmetric function，比如取最大值）去聚合所有点的特征，就能天然满足这个不变性。这个洞察构成了整篇论文的理论基石，实现起来却异常简单——对每个点独立地跑一个共享的 MLP，再对所有点的输出做一次 max pooling。

除了排列不变性，论文还引入了 **T-Net**（Transformation Network）：一个小号的 PointNet，从数据本身回归出一个仿射变换矩阵，分别作用在输入坐标（3x3）和中间特征（64x64）上，让模型对点云的刚体变换（旋转、平移）更鲁棒。

PointNet 最大的局限是"全局 max pooling"这一步——它把整个点云的所有点一次性聚合成一个全局特征，完全没有局部邻域的概念，这与 CNN 通过层层卷积构建局部到全局的层次化感受野的思路是相反的。同年（NeurIPS 2017）同一个团队推出了 [**PointNet++**](https://arxiv.org/abs/1706.02413)，引入了分层的 set abstraction：通过最远点采样（FPS）和球查询（ball query）划出局部邻域，再递归地在每个局部邻域里跑一个 mini-PointNet，从而补上了局部结构建模的短板。

## 二、模型结构，以及转换难度分析

PointNet 的骨架（`PointNetEncoder`）大致是这样的：

1. **输入**：`[B, C, N]`，B 是 batch，C 是每个点的通道数（最简单的情况是 xyz 三通道，语义分割任务里常见的是 xyz + rgb + 归一化 xyz 共 9 通道），N 是点数。
2. **STN3d（输入变换）**：3 层共享 MLP（用 kernel_size=1 的 Conv1d 实现，等价于对每个点独立做全连接）+ BN + ReLU，然后在点维度上做一次 **global max pool**，接两层 FC，最后回归出一个 3x3 矩阵，并加上单位矩阵作为偏置（让 T-Net 初始状态接近恒等变换，训练更稳定）。这个矩阵通过 **`torch.bmm`**（batched matrix multiply）作用在输入点云上。
3. **共享 MLP 特征提取**：继续用 Conv1d(kernel=1) 把每个点的特征从低维升到 64 维、再到 1024 维。
4. **STNkd（特征变换，64x64）**：跟 STN3d 结构一样，只是维度换成了 64，同样通过 `bmm` 作用在中间特征上。
5. **对称函数（全局 max pool）**：`torch.max(x, dim=points维度, keepdim=True)`，把 N 个点的特征聚合成一个 1024 维全局特征向量——这就是保证排列不变性的关键操作。
6. **分割头 vs 分类头**：
   - 分类任务：只用这个全局特征过几层 FC 输出类别 logits。
   - 分割任务：把全局特征 broadcast/repeat 回 N 个点，跟每个点自己的局部特征 **concat** 起来（1024+64=1088 维），再过几层 Conv1d(kernel=1) 输出每个点的类别 logits，最后 `log_softmax`。

**转换难度怎么样？** 结论是：**几乎没有难度。**

- `Conv1d(kernel_size=1)`、`BatchNorm1d`、`Linear`、`ReLU`、`transpose`、`cat`、`log_softmax` 都是 OpenVINO 转换器原生支持得很好的算子，不会遇到需要 fallback 或自定义算子实现的情况。
- 唯二两个"看起来特殊"的操作——`torch.bmm`（用于施加 T-Net 变换矩阵）和 `torch.max(dim=N, keepdim=True)`（全局对称聚合，注意这个 max 是在点数这个"类似 sequence"的维度上做的，不是空间卷积里常见的 2D max pooling）——OpenVINO 都能直接对应到自己的 MatMul 和 ReduceMax 算子，不需要图替换或算子拆解。

所以 PointNet 是名副其实的"转换友好"模型：没有自定义 CUDA 算子，没有需要图手术的特殊结构，所有算子在 OpenVINO 的算子集里都能找到一一对应的实现。这也是为什么下一节的转换流程会显得格外简单。

## 三、OpenVINO 怎么转换的，需要做什么模型修改

### 3.1 不走 ONNX，直接 PyTorch → OpenVINO，核心代码只有几行

这次转换完全没有生成中间的 `.onnx` 文件，而是用 OpenVINO 的 PyTorch 前端直接对 `nn.Module` 做转换（`ov.convert_model`，底层基于 TorchScript/FX 追踪）。之所以能这样做，正是因为上一节说的——所有算子 OpenVINO 都原生支持，不需要借助 ONNX 生态的算子转换/simplify 工具去做兼容处理。核心转换代码就是这几行：

```python
model = PointNetSemSegOV(num_classes=13)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 9, 4096)
ov_model = ov.convert_model(model, example_input=dummy_input, input=[(1, 9, 4096)])
ov.save_model(ov_model, "pointnet_sem_seg_simplified.xml", compress_to_fp16=True)
```

加载权重、`eval()`、给一个示例输入、调用 `ov.convert_model`、`save_model` 落盘——一气呵成，中间不需要任何额外的算子替换或图重写步骤。

### 3.2 唯一需要的"模型修改"：包一层壳，去掉训练专用输出

上游实现里，`forward()` 返回的是一个 tuple：`(seg_logits, trans_feat)`。`trans_feat` 是训练时用来算特征变换矩阵正交正则化损失的，推理阶段完全没用。但 `ov.convert_model` 更适合处理单一 tensor 输出的干净计算图，所以我们写了一个薄薄的 wrapper：

```python
class PointNetSemSegOV(nn.Module):
    """Wrap PointNet so OpenVINO conversion exports only segmentation logits."""
    def __init__(self, num_classes=13):
        super().__init__()
        self.model = get_model(num_class=num_classes)

    def forward(self, x):
        logits, _ = self.model(x)
        return logits
```

注意：这不是"图手术"（没有替换、删除或重写任何算子），上游源码原封不动，纯粹是在调用层面砍掉一个用不上的输出。

**这层壳会不会带来性能提升？** 我们实际做了对比实验：分别用带 wrapper（单输出）和不带 wrapper（tuple 双输出，`logits` + `trans_feat`）两种方式转换同一份权重，得到两份 IR，再用 `benchmark_app` 在 GPU 上各跑两轮（`-infer_precision f16`，15s/轮）：

| 版本 | 图中算子数 | 输出数 | FPS（两轮） | 中位延迟（两轮） |
|---|---|---|---|---|
| 不带 wrapper（tuple 输出） | 184 | 2 | 584.44 / 578.94 | 6.81ms / 6.88ms |
| 带 wrapper（单输出） | 183 | 1 | 582.93 / 581.46 | 6.83ms / 6.85ms |

结论是：**吞吐量基本没有差异**，两个版本的数字都落在同一个噪声区间里。原因也不难理解——`trans_feat` 本来就是 64x64 feature-transform T-Net 算出来的中间结果（施加 `bmm` 之前必须先算出这个矩阵），去掉它作为输出只是少了一次"把这个 tensor 拷贝到输出"的操作，对应到图里正好是 184 个算子减到 183 个，并没有省掉任何实际计算。所以这个 wrapper 的价值主要是**让导出的计算图更干净、下游推理代码不用处理无用的第二个输出**，写起来也是几行代码的事。

### 3.3 转换后的数值验证方法

验证脚本对同一批固定种子的输入（1 个结构化的类 S3DIS 点云 + 12 个不同分布的合成点云），跑三条推理管线做两两比对：

- PyTorch CPU FP32（基准）
- PyTorch CPU FP16（autocast，而不是 `.half()`——因为 T-Net 里 `np.eye(...)` 生成的单位矩阵常量在强制类型转换下会出问题）
- OpenVINO GPU FP16（IR 推理结果）

比较时统一把 `log_softmax` 的输出转成概率空间（`exp()`）再算差值，因为 log 值在模型很自信地拒绝某个类别时会跌到 -1000 量级，直接比较原始 log 值会让所有误差指标被这些无意义的尾部数值主导。最终验证结果：两组比较（OV-GPU vs Torch-CPU-FP16 的严格比较，以及 OV-GPU vs Torch-CPU-FP32 的宽松比较）在全部 13 个输入上都通过，没有 NaN，最差情况下 top-1 匹配率也在 99.6% 以上。

最终 benchmark 结果：CPU 75.35 FPS（12.97ms 中位延迟），GPU 558.91 FPS（1.76ms 中位延迟），约 7.4 倍加速。

## 四、相关仓库与资料

- **PointNet 原始论文**：[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)（CVPR 2017）
- **PointNet++ 论文**：[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)（NeurIPS 2017）
- **本文使用的 PyTorch 实现（上游，作为 git submodule 引入，未做任何修改）**：[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- **本文的转换代码仓库**：[zyz9740/export_models](https://github.com/zyz9740/export_models)，PointNet 相关的转换脚本、权重下载、验证框架都在 `export_pointnet/` 目录下：
  - `converter/model.py`、`converter/convert.py`：模型 wrapper 与转换脚本
  - `validation/validate.py`、`validation/validation_report.md`：跨设备数值验证脚本与报告
  - `benchmark/`：CPU/GPU benchmark 日志
  - `demo/infer_demo.py`：端到端推理示例
- **OpenVINO 官方文档**：[docs.openvino.ai](https://docs.openvino.ai/)，PyTorch 直转 API 参考 [Converting a PyTorch Model](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-pytorch.html)
- **数据集 S3DIS**：本文用到的语义分割权重训练自 Stanford 3D Indoor Scene Dataset（13 类室内场景标注）

总的来说，PointNet 是一个对 OpenVINO 非常友好的模型：不需要 ONNX 中转，不需要图手术，也不需要处理自定义算子，从加载权重到落地 IR 文件，核心代码不到十行就能跑通，剩下要做的只是按需包一层壳把输出整理干净。如果你也在用 OpenVINO 部署点云模型，PointNet 这条转换路径可以直接拿来当模板。
