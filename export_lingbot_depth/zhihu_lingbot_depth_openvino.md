# 用 OpenVINO 跑 LingBot-Depth：一个 1.2GB 的 ViT 深度大模型，转换要动哪几刀

上一篇写 [PointNet 转 OpenVINO](https://github.com/zyz9740/export_models) 时，结论是"几乎没有难度"——所有算子原生支持，核心代码不到十行。这次换一个"重量级"的模型：蚂蚁团队 2026 年放出的 **LingBot-Depth**，一个基于 DINOv2 ViT-Large 的 RGB-D 深度修复大模型，checkpoint 有 1.2GB，671 个 IR 算子。它就没那么"温柔"了——转换过程中真的踩到了 OpenVINO 不认识的算子，需要动手术。这篇文章讲清楚它是什么、结构长什么样、转换时到底动了哪几刀，最后附上一份 GPU 上的性能剖析，看看这类 transformer 模型的时间都花在哪。

## 一、LingBot-Depth 是什么

LingBot-Depth 出自蚂蚁（Robbyant/AntGroup）团队，论文题目是 [《Masked Depth Modeling for Spatial Perception》](https://arxiv.org/abs/2601.17895)，已被 ECCV 2026 接收。

它解决的问题是**深度修复与补全**：消费级深度相机（Intel RealSense、Orbbec Gemini、Azure Kinect 这些）拍出来的原始深度图往往是残缺、带噪声的——反光表面、透明物体、远距离区域经常是一片空洞或者乱跳的值。LingBot-Depth 把「RGB 图像 + 残缺的原始深度」一起喂进去，输出一张**干净、完整、保留真实物理尺度（metric-accurate）**的深度图，再配合相机内参就能反算出高质量的三维点云。

它的定位是机器人和 3D 视觉的**空间感知基座模型**，下游能直接接抓取（dexterous grasping）、4D 点追踪、场景重建这些任务。训练数据规模也很夸张——300 万张 RGB-D，其中 200 万真实采集 + 100 万仿真渲染。

核心方法叫 **Masked Depth Modeling（掩码深度建模）**：借鉴 MAE 的自监督思路，训练时随机遮住一部分深度、让模型去重建，从而学到 RGB 外观和深度几何在同一潜空间里的对齐关系。骨干网络用的是 **DINOv2 ViT-Large**，但改造成了能同时吃 RGB 和 depth 两路输入的 **RGBD encoder**，中间靠跨模态注意力把两种模态融合起来。这个思路很像我刚上大学的时候听到的 Bert 的训练思路，把深度图的一部分遮挡起来和把句子中的一些词遮挡起来的思路很像。

一句话总结：**这是一个 ViT-Large 尺度的 encoder-decoder 大模型**，跟 PointNet 那种几 MB 的轻量网络完全不是一个量级。这也预告了转换难度的差异。

## 二、模型结构，以及转换难度分析

从 `MDMModel`（`mdm/model/v2.py`）看，整个前向大致是这样一条链：

1. **输入**：`image` `[B, 3, H, W]`（归一化到 [0,1]）+ `depth` `[B, H, W]`（单位米，无效区域为 0 或 NaN）。这次导出用的是 `[1, 3, 480, 640]` 和 `[1, 480, 640]` 的静态形状，对应 example 场景 0。
2. **DINOv2 RGBD Encoder**：把 image 和 depth 分别 patchify（patch=14），送进 24 层 ViT block，每层是标准的 `qkv 投影 → scaled_dot_product_attention → MLP`，RGB 与 depth token 之间做跨模态注意力融合。这是整个模型**最重**的部分。
3. **Neck（ConvStack）**：把 encoder 出来的 token 特征 reshape 回空间特征图，再拼上归一化的视平面 UV 坐标（用来编码宽高比/相机视角），做多尺度特征金字塔。
4. **多个 Head**：`depth_head` 回归深度、`mask_head` 出有效区域掩码。都是 ConvStack + 上采样。
5. **输出重映射**：深度经过 `exp()` 从 log 空间映射回米制尺度；mask 过 `sigmoid`；最后 `F.interpolate` 双线性插值放大回原图分辨率 `[B, H, W]`。
6. **点云生成**：`depth_to_pointcloud(depth, intrinsics)`——用相机内参把每个像素的深度反投影成相机坐标系下的 `[B, H, W, 3]` 点云。

**转换难度怎么样？** 结论是：**比 PointNet 麻烦，但麻烦点很集中——两处图手术 + 一层输出裁剪，都是"改导出路径、不改上游源码"的干净做法。**

大头其实是没问题的：ViT 里的 `Linear`（qkv/mlp）、`LayerNorm`、`scaled_dot_product_attention`、`Conv`、`interpolate` 这些 OpenVINO 都原生支持得很好，SDPA 甚至直接映射到了优化过的 `ocl::sdpa::opt` 融合 kernel。真正卡住转换的是下面两个点。

## 三、OpenVINO 怎么转换的，动了哪几刀

跟 PointNet 一样，这次也是**不走 ONNX**，直接用 OpenVINO 的 PyTorch 前端对 `nn.Module` 做 `ov.convert_model`。但要让 `convert_model` 顺利跑通，得先解决三件事。

### 3.1 第一刀（真·图手术）：替换 `aten::nan_to_num`

这是唯一一个**导致转换直接报错**的算子。原始深度图里无效区域是 NaN/Inf，encoder 里用 `torch.nan_to_num(depth_14, nan=0.0, posinf=0.0, neginf=0.0)` 把它们清零。但 OpenVINO 的 PyTorch 前端追踪到 `aten::nan_to_num` 时会失败。

解法是在导出阶段用一个等价实现替换掉它——`where(isfinite(x), x, 0)`：

```python
def patch_nan_to_num_for_export():
    def export_friendly_nan_to_num(input_tensor, nan=0.0, posinf=None, neginf=None, out=None):
        replacement = torch.full_like(input_tensor, float(nan))
        return torch.where(torch.isfinite(input_tensor), input_tensor, replacement)
    torch.nan_to_num = export_friendly_nan_to_num
```

`isfinite`、`where`、`full_like` 都是 OpenVINO 认识的算子，语义上跟 `nan_to_num(nan=0)` 完全等价（把非有限值替换成 0），无效深度清理的行为一点没变。**注意这是在导出脚本里 monkey-patch `torch.nan_to_num`，上游 `lingbot-depth` 源码原封不动**——这才是真正意义的"图手术"，跟 PointNet 那种"只是包个壳"不一样。

**这个补丁在 IR 里到底长什么样？** 去 `lingbot_depth.xml` 里顺着这条替换路径查一遍会发现：它不会收缩成一个算子，而是变成一个 **5 层子图**——因为 `where(isfinite(x), x, full_like(x, 0))` 本来就是三个独立的 PyTorch 算子，每个都各自 1:1 映射到一个 OpenVINO 层：

| IR layer id | 名称 | 类型（opset） | 作用 |
|---|---|---|---|
| 99 | `aten::isfinite/IsFinite` | `IsFinite`（opset10） | `torch.isfinite(x)`——算出 BOOL 条件掩码 |
| 100 | `495_compressed` | `Const`（opset1） | `nan=0.0` 这个标量常量，以 FP16 存储 |
| 101 | `495` | `Convert`（opset1） | 把上面的 FP16 常量解压回 FP32 |
| 102 | `aten::full_like/ShapeOf` | `ShapeOf`（opset3） | 读取输入 `x` 的形状，为下一步广播做准备 |
| 103 | `aten::full_like/Broadcast` | `Broadcast`（opset3） | `torch.full_like(x, 0)`——把标量广播成 `x` 的形状，就是 `replacement` |
| 104 | `aten::where/Select` | `Select`（opset1） | `torch.where(...)`——OpenVINO 没有专门的 `Where` 算子，`aten::where` 一律降级成 `Select` |

从 XML 里的 `<edge>` 连线看，这几层是这样接起来的：

```
layer 99 (IsFinite)         <- x               [算条件]
layer 100 (Const f16=0.0) -> layer 101 (Convert f16->f32)
layer 101 -> layer 103 端口0   (Broadcast: 标量值)
layer 102 (ShapeOf, 读 x 的形状) -> layer 103 端口1  (Broadcast: 目标形状)
layer 99  -> layer 104 端口0   (Select: 条件)
x         -> layer 104 端口1   (Select: 条件为真时取值)
layer 103 -> layer 104 端口2   (Select: 条件为假时取值，即 0)
layer 104 (Select) -> 继续喂给 encoder 下游
```

也就是说 layer 104 的 `Select(condition=IsFinite(x), then=x, else=Broadcast(0.0))` 就是 `nan_to_num(x, nan=0.0)` 在 OpenVINO 里的原生等价实现，作用在一张 `[1, 1, 420, 560]` 的深度特征图上（对应本次 480x640 输入、按 patch=14 下采样后的深度图）。整个子图里没有一个自定义算子或 fallback 层——全是标准 opset 算子，这正是这个补丁能生效的原因：把追踪不了的 `aten::nan_to_num` 改写成了 OpenVINO 本来就会原生降级的几个算子组合。

### 3.2 第二刀：打开 `onnx_compatible_mode` 关掉 antialias

encoder 里对图像做 `F.interpolate(..., antialias=True)`。抗锯齿插值在导出场景里同样是个麻烦算子。好在上游作者自己留了一个开关 `onnx_compatible_mode`，打开后 `antialias` 会自动置为 `False`：

```python
model.encoder.onnx_compatible_mode = True
# 源码里：antialias = not self.onnx_compatible_mode
```

这一刀甚至不用我们自己写——上游为了兼容 ONNX/导出场景本来就预留了这条路径，我们只要把开关拨过去。对推理结果的影响仅限于插值时是否做抗锯齿，对深度精度无实质影响。

### 3.3 第三刀（包壳）：固定动态量、裁剪输出

`MDMModel.forward` 有几个对静态图不友好的地方：`num_tokens` 参数会参与算 `base_h/base_w`（决定 ViT 的 token 网格大小），返回值是个 dict 且带训练用的额外字段。跟 PointNet 一样，包一层薄壳把它整理干净：

```python
class LingBotDepthExportWrapper(torch.nn.Module):
    def __init__(self, model, num_tokens):
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)   # 固定成常量，静态图不再依赖它

    def forward(self, image, depth):
        output = self.model.forward(
            image=image, num_tokens=self.num_tokens,
            depth=depth, enable_depth_mask=False,
        )
        depth_reg = output["depth_reg"]
        mask = output.get("mask")
        if mask is None:
            mask = torch.ones_like(depth_reg, dtype=torch.bool)
        return depth_reg, mask.to(dtype=depth_reg.dtype)   # 只导出 depth + mask 两个 tensor
```

这里做了三件事：把 `num_tokens` 固定成常量（本次用 1200）、`enable_depth_mask=False` 让计算图保持静态张量形状、把 dict 输出拍平成 `(depth, mask)` 两个纯 tensor。

### 3.4 核心转换代码

三刀备好之后，转换本身还是很短：

```python
os.environ.setdefault("XFORMERS_DISABLED", "1")   # 禁掉 xformers，走标准 SDPA
torch.set_grad_enabled(False)
patch_nan_to_num_for_export()                     # 第一刀

model = MDMModel.from_pretrained(model_id_or_path).eval().cpu()
model.encoder.onnx_compatible_mode = True         # 第二刀

wrapped = LingBotDepthExportWrapper(model, num_tokens=1200).eval()   # 第三刀
image = torch.zeros((1, 3, 480, 640), dtype=torch.float32)
depth = torch.ones((1, 480, 640), dtype=torch.float32)

ov_model = ov.convert_model(
    wrapped,
    example_input=(image, depth),
    input=[
        ("image", [1, 3, 480, 640], ov.Type.f32),
        ("depth", [1, 480, 640], ov.Type.f32),
    ],
)
ov_model.outputs[0].get_tensor().set_names({"depth"})
ov_model.outputs[1].get_tensor().set_names({"mask"})
ov.save_model(ov_model, "lingbot_depth.xml", compress_to_fp16=True)
```

落盘的 IR 是 `lingbot_depth.xml`（约 1.4MB 的图结构）+ `lingbot_depth.bin`（约 612MB 的 FP16 权重）。

### 3.5 点云生成放在图外面

有一点值得单独说：模型 IR **只导出 `depth` 和 `mask`**，点云不在图里。原因是 `depth_to_pointcloud` 依赖相机内参 `intrinsics`，这是每帧/每台相机都可能变的运行期参数，硬编进静态图不合适。所以点云反投影放在 demo 的**后处理**里用 NumPy 做——拿 OpenVINO 出来的深度图 + 内参，`x=(u-cx)*z/fx, y=(v-cy)*z/fy, z=depth` 三行算出 `[H, W, 3]` 点云。这跟 PointNet 那篇里"分割 logits 全在图内"的处理不同，是这类需要相机参数的几何模型的常见做法。

## 四、转换后的数值验证

验证脚本对 10 组固定种子的随机输入（`image` 随机、`depth` 在 0.1~4.1 米范围随机），比对 **PyTorch CPU** 与 **OpenVINO GPU FP16** 两条管线的 `depth` / `mask` 输出。

结果：**10 组输入全部无 NaN**，深度输出的 `max_abs` 差异落在 0.093~0.250、`mean_abs` 差异落在 0.035~0.081 的区间——对一个深度值本身在米级、且走了 FP16 压缩路径的模型来说，这个量级的偏差属于 FP16 执行路径的正常范围，判定为通过。

（这里的验证比 PointNet 那篇宽松一些：PointNet 是分类 logits，可以直接看 top-1 匹配率；深度是连续回归量，看的是绝对/平均误差是否在 FP16 噪声带内。）

## 五、GPU 性能剖析：时间都花在哪

这个模型在 Intel Core Ultra 265H 的 iGPU Intel Arc 140T GPU（16GB，128 XVE）上 `benchmark_app -hint latency -infer_precision f16` 的成绩是**中位延迟 202.51ms，吞吐 4.89 FPS**。ViT-Large 尺度的模型，单帧两百毫秒符合预期。

用 `-exec_graph_path` 抓了逐层耗时，再叠加一次 VTune GPU Hotspots，结论很清晰——**时间几乎全在 transformer 骨干上**：

| 算子类型 | 占比 | 层数 | 说明 |
|---|---|---|---|
| **FullyConnected** | 35.9% | 96 | ViT 的 qkv / mlp 线性投影，走优化 `jit:gemm` kernel |
| **scaled_dot_product_attention** | 32.1% | 24 | 24 个 ViT block 的注意力，走优化 `ocl::sdpa::opt` kernel |
| Pad | 8.3% | 36 | ⚠️ 全部落在**参考实现** `border_gpu_ref` |
| Convolution | 8.1% | 52 | neck / head 的卷积 |
| Permute | 4.6% | 49 | 注意力里的 reshape/transpose，部分落在参考实现 |
| 其余（MVN/Reorder/Resample…） | ~11% | — | |

**FullyConnected + SDPA 合计约 68%**，这是 ViT transformer 的本分，跑在优化 kernel 上，不是问题。

真正有意思的优化信号是那 **13% 花在参考路径（`_ref`）kernel 上的 `Pad` 和 `Permute`**——padding 和转置本质是纯访存操作，没有任何算法理由跑得慢，却因为 OpenVINO GPU 插件没给它们选到优化 kernel 而占了 13% 的时间。三个最贵的 Pad 都是 neck/depth_head/mask_head 上采样路径末端、在最大空间分辨率上做的 border padding。这是后续做图级优化（换 kernel、或重排 pad 让它融进相邻 conv）最值得下手的地方。

VTune 侧还有个观察：整个 run **XVE 阵列 84.1% 处于 stalled/idle，占用率仅 36.9%，XMX 时间加权活跃度只有 3.23%**。也就是说这个 workload 是**访存/调度/布局受限**，而不是被 XMX 矩阵计算撑满——这也解释了为什么"纯访存"的 Pad/Permute 参考 kernel 会成为最显眼的短板。

## 六、相关仓库与资料

- **LingBot-Depth 论文**：[Masked Depth Modeling for Spatial Perception](https://arxiv.org/abs/2601.17895)（ECCV 2026）
- **上游实现（作为 git submodule 引入，未做任何修改）**：[Robbyant/lingbot-depth](https://github.com/Robbyant/lingbot-depth)
- **推荐权重**：[robbyant/lingbot-depth-pretrain-vitl-14-v0.5](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5)（HuggingFace）/ [ModelScope 镜像](https://www.modelscope.cn/models/Robbyant/lingbot-depth-pretrain-vitl-14-v0.5)
- **DINOv2 骨干**：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **本文的转换代码仓库**：[zyz9740/export_models](https://github.com/zyz9740/export_models)，LingBot-Depth 相关内容都在 `export_lingbot_depth/` 目录下：
  - `converter/convert.py`：模型 wrapper、`nan_to_num` 替换与转换脚本
  - `validation/validate.py`、`validation/validation_report.md`：跨设备数值验证脚本与报告
  - `validation/profiling/`：逐层 GPU 剖析（`gpu_layer_profile_report.md`）与 VTune XMX 分析（`vtune_xmx_analysis_report.md`）
  - `demo/infer_demo.py`：端到端推理示例（含点云后处理）
- **OpenVINO 官方文档**：[Converting a PyTorch Model](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-pytorch.html)

## 结语

跟 PointNet 那篇的"转换友好模板"对照着看，LingBot-Depth 给出了另一个更真实的场景：**一个 1.2GB 的 ViT 大模型，绝大多数算子仍然原生支持，真正需要动手的只是几个明确的点**——一处 `nan_to_num` 图手术、一个上游预留的 `onnx_compatible_mode` 开关、一层固定动态量和裁剪输出的壳，加上把依赖相机内参的点云生成留在图外做后处理。核心思路没变：**能不改上游源码就不改，所有修改都收敛在导出脚本里**。转换跑通之后，profiling 又告诉我们瓶颈不在注意力本身，而在两个"不该慢"的访存算子上——这才是下一步优化真正该盯的地方。如果你要在 Intel GPU 上部署这类 RGB-D transformer 深度模型，这条路径可以直接参考。
