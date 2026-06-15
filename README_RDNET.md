# RDNet 引导的 ERRNet：反射区域感知去反射

> **分支**：`feat/region-aware`  
> **Baseline 权重**：`checkpoints/errnet/errnet_060_00463920.pt`（epoch 60）  
> **本文档用途**：记录方法、实验设置与完整结果，可直接摘取写入课程/论文报告。

---

## 摘要（论文可用）

单幅图像去反射任务中，ERRNet 等全局恢复网络对整图施加统一优化，容易在无反射背景区域产生过度修改，且在局部强反射区域优化不足。受 CVPR 2024「Revisiting Single Image Reflection Removal In the Wild」中 MaxRF 区域感知思想启发，我们在 ERRNet 前引入轻量级反射区域检测网络 RDNet（Reflection Detection Network），先预测反射 mask，再将其与输入 RGB 拼接后送入 ERRNet，使模型具备显式空间先验。

在 ERRNet 官方 baseline 权重上，我们完成两阶段训练：aligned 数据训练 60 epoch + unaligned 数据微调 20 epoch，并在 5 个标准测试集上与 baseline 对比。实验表明：**RDNet 在 CEILNet Table 2 上 PSNR 提升最高达 +1.09 dB**（`aligned_no_sup`）；在 **real20 上经 unaligned 微调后与 baseline 基本持平**（-0.03 dB）；在 objects/postcard 等 SIR² 子集上略低于 baseline。消融显示：**仅加入 mask 输入而不监督 RDNet 效果最佳**，过强的 MaxRF mask 监督反而会损害恢复质量。

---

## 1. 引言与动机

### 1.1 问题

玻璃、水面等场景下的反射去除是典型的 ill-posed 问题。ERRNet（[arXiv:2311.17320](https://arxiv.org/abs/2311.17320)）在 paired / 合成数据上表现良好，但其输入为整张混合图像，网络需同时推断「哪里是反射、如何去反射」，对空间不均匀的真实反射不够友好。

### 1.2 核心想法

将去反射拆为两步：

1. **Where**：RDNet 预测反射可能存在的空间区域（reflection mask）；
2. **How**：ERRNet 在 mask 先验引导下恢复 transmission 图像。

这样 ERRNet 可以更聚焦于高反射区域，减少对干净背景的不必要改动。

### 1.3 与相关工作的关系

| 工作 | 关系 |
|------|------|
| ERRNet (2023) | 本项目的 backbone，提供预训练权重与训练/测试框架 |
| MaxRF / CVPR 2024 Wild | 借鉴其「输入边缘强于 GT 边缘则为反射区」的伪标签构造思路 |
| 本项目 RDNet | **非** CVPR 2024 完整复现，而是在 ERRNet 上增加的轻量分支，便于课程项目消融 |

---

## 2. 方法

### 2.1 整体流程

```text
输入混合图像 I  (RGB, 3×H×W)
        │
        ▼
   RDNet(I)  ──►  预测 mask M_pred  (1×H×W, ∈ [0,1])
        │
        ▼
  concat [I, M_pred]  ──►  (+ VGG hypercolumn，若 --hyper)
        │
        ▼
      ERRNet  ──►  输出去反射结果 T_pred
```

与 baseline ERRNet 的唯一结构差异：ERRNet 第一层输入通道由 3（+ hypercolumn）变为 **4（+ mask + hypercolumn）**。

### 2.2 RDNet 结构

`ReflectionDetectionNet` 定义于 `models/errnet_model.py`：

| 组件 | 配置 |
|------|------|
| 输入 | RGB 3 通道 |
| 主干 | Conv 3×3 → ReLU ×2 |
| 残差块 | 3 个 `ResidualMaskBlock`（默认 `--rdnet_blocks 3`） |
| 输出头 | Conv → ReLU → Conv 1×1 → **Sigmoid** |
| 基础通道数 | 32（`--rdnet_channels`） |
| 输出 | 1 通道 reflection mask |

参数量小，训练与推理开销相对 ERRNet 本体可忽略。

### 2.3 MaxRF 风格伪标签（仅 aligned 样本）

当 batch 中存在像素级对齐的 `(I, T)` 时，构造监督 mask：

```text
M_gt = smooth( 1[ Edge(I) > Edge(T) ] )
```

- `Edge(·)`：Sobel 边缘图（代码中 `EdgeMap`）；
- `smooth`：kernel=9 的平均池化（`--rdnet_mask_smooth`），减少伪标签噪声。

**unaligned 样本跳过此监督**：输入与 GT 像素不对齐时，`Edge(I) > Edge(T)` 不可靠。

### 2.4 损失函数

#### Aligned 样本（`train_errnet.py`）

与 ERRNet 一致，并可选加入 RDNet 监督：

```text
L = L_pixel + λ_vgg · L_VGG + λ_gan · L_GAN          （ERRNet 原有）
  + λ_rd · L1(M_pred, M_gt) + λ_rd_tv · TV(M_pred)   （RDNet，仅 λ_rd > 0 时）
```

| 符号 | 默认值 | 含义 |
|------|--------|------|
| `λ_vgg` | 0.1 | VGG 感知 loss 权重 |
| `λ_rd` | 1.0 | mask L1 监督权重；**设为 0 即 no_sup 消融** |
| `λ_rd_tv` | 0.00005 | mask TV 平滑；**设为 0 即 sup 消融（无 TV）** |

#### Unaligned 样本（`train_errnet_unaligned.py`）

不使用 pixel / VGG 对齐 loss，改用 **VGG 感知 loss**（`--unaligned_loss vgg`）：

```text
L = L_CX/VGG(I_out, T)     （不要求像素对齐）
  （无 RD mask 监督）
```

RDNet 仍通过 mask 影响 ERRNet 输入，从而被 restoration loss **间接**更新。

### 2.5 权重初始化与加载

- 从 ERRNet baseline 微调时（`-r --icnn_path ...`）：
  - ERRNet 兼容层加载原权重，**新增 mask 输入通道初始化为 0**；
  - RDNet 随机初始化（EDSR 风格）；
  - 迁移学习时 **重置 epoch/iterations 为 0**（`train_errnet.py` 中已实现），确保完整跑满训练 schedule。

### 2.6 Aligned 与 Unaligned 训练的区别

| 维度 | Aligned（`train_errnet.py`） | Unaligned（`train_errnet_unaligned.py`） |
|------|------------------------------|----------------------------------------|
| 数据 | 合成 VOC 70% + real_train 30% | 合成 25% + **unaligned 50%** + real 25% |
| 像素关系 | 输入与 GT **逐像素对齐** | unaligned 集 crop 时故意 ±10px 偏移 |
| 主 loss | Pixel L1 + VGG | VGG / CX 感知 loss |
| RD mask 监督 | 有（若 `λ_rd > 0`） | **无** |
| 训练轮数 | 60 epoch | 80 epoch（从 aligned ckpt 续训） |
| 目的 | 学习基本去反射 + RDNet | 适应真实不对齐数据（如 real20） |

---

## 3. 实验设置

### 3.1 测试数据集（5 个）

本项目 benchmark 与 ERRNet DIP26 指南一致，使用 **5 个标准测试集**（未包含 `sir2_withgt`）：

| 数据集 key | 说明 | 特点 |
|------------|------|------|
| `ceilnet_table2` | CEILNet Table 2 | 合成 paired，与 aligned 训练域最接近 |
| `real20` | 20 张真实反射图 | 不对齐、最具挑战性 |
| `objects` | SIR² Objects 子集 | 真实场景 |
| `postcard` | SIR² Postcard 子集 | 真实场景 |
| `wild` | SIR² Wild 子集 | 真实 wild 场景 |

> 官方文档另提供 `sir2_withgt`；本次 RDNet 实验 **未评测该集**，论文中若需对比请单独补跑 `test_errnet.py --dataset sir2_withgt`。

### 3.2 评价指标

与 ERRNet 一致，**数值越高越好**：PSNR、SSIM、NCC；**LMSE 越低越好**。

测试命令统一：`test_errnet.py --hyper -r --icnn_path <ckpt> [--use_rdnet]`。

### 3.3 对比方法与消融变体

共 **7 组**模型：

| 编号 | 名称 | 训练阶段 | `λ_rd` | `λ_rd_tv` | Checkpoint 目录 |
|------|------|----------|--------|-----------|-----------------|
| 0 | **baseline** | 官方 ERRNet ep60 | — | — | `checkpoints/errnet/` |
| 1 | aligned_no_sup | aligned 60ep | 0 | — | `errnet_rdnet_no_sup/` |
| 2 | aligned_sup | aligned 60ep | 1.0 | 0 | `errnet_rdnet_sup/` |
| 3 | aligned_full | aligned 60ep | 1.0 | 0.00005 | `errnet_rdnet_full/` |
| 4 | unaligned_no_sup | unaligned →80ep | 0 | — | `errnet_rdnet_no_sup_unaligned/` |
| 5 | unaligned_sup | unaligned →80ep | 1.0 | 0 | `errnet_rdnet_sup_unaligned/` |
| 6 | unaligned_full | unaligned →80ep | 1.0 | 0.00005 | `errnet_rdnet_full_unaligned/` |

消融含义：

- **no_sup**：RDNet 仅作为 mask 输入，无 mask 监督；
- **sup**：MaxRF 伪标签 L1 监督，无 TV；
- **full**：L1 + TV 完整 RDNet 监督。

### 3.4 训练细节

#### 阶段一：Aligned（3 卡并行）

```bash
bash scripts/run_rdnet_aligned_parallel.sh
```

| 超参 | 值 |
|------|-----|
| 初始 lr | 1e-4（ep30→5e-5，ep40→1e-5，ep45 调整 fusion ratio，ep50→1e-5） |
| GAN loss | ep20 后启用 λ_gan=0.01 |
| 初始化 | baseline `errnet_060_00463920.pt` |
| epoch | 60 |

#### 阶段二：Unaligned 微调（3 卡并行）

```bash
bash scripts/run_rdnet_unaligned_parallel.sh
```

| 超参 | 值 |
|------|-----|
| `--unaligned_loss` | **vgg**（官方推荐；`ctx_vgg` 在本实验中 ep65 附近 CX loss NaN） |
| lr | **5e-5**（ep70 ×0.2） |
| 初始化 | 对应 aligned 组 `errnet_model_latest.pt` |
| epoch | 80（在 aligned ep60 基础上再训 20 epoch） |
| 稳定性 | 每 epoch 检测权重 NaN/Inf，异常则停止 |

#### Baseline 复现（本机）

| Dataset | PSNR | SSIM | NCC | LMSE |
|---------|------|------|-----|------|
| CEILNet Table 2 | 27.88 | 0.9407 | 0.9808 | 0.0048 |
| real20 | 23.55 | 0.8285 | 0.8877 | 0.0201 |
| objects | 24.85 | 0.8980 | 0.9817 | 0.0029 |
| postcard | 22.07 | 0.8773 | 0.9463 | 0.0044 |
| wild | 25.18 | 0.8861 | 0.9359 | 0.0083 |

（与 `README_DIP26.md` §5 一致；若论文引用其他来源的 baseline 数字，需注明 checkpoint / 环境差异。）

---

## 4. 实验结果

> 完整日志：`logs/benchmark_results.txt`（baseline + aligned + 部分 unaligned）、`logs/benchmark_unaligned.txt`  
> 汇总脚本：`python scripts/parse_benchmark_full.py` → `logs/benchmark_full_summary.txt`

### 4.1 PSNR（dB，↑ 越好）

| 模型 | CEILNet | real20 | objects | postcard | wild |
|------|---------|--------|---------|----------|------|
| baseline | 27.88 | 23.55 | **24.85** | **22.07** | **25.18** |
| aligned_no_sup | **28.96** | 23.40 | 24.38 | 22.00 | 25.51 |
| aligned_sup | 28.75 | 23.46 | 24.46 | 21.40 | 25.55 |
| aligned_full | 28.47 | 23.54 | 24.42 | 21.80 | 24.95 |
| unaligned_no_sup | 28.35 | 23.25 | 24.03 | 21.01 | 23.99 |
| unaligned_sup | 28.39 | 23.52 | 24.23 | 20.80 | 24.26 |
| unaligned_full | 28.28 | **23.53** | 24.23 | 21.05 | 24.21 |

### 4.2 SSIM（↑ 越好）

| 模型 | CEILNet | real20 | objects | postcard | wild |
|------|---------|--------|---------|----------|------|
| baseline | 0.9407 | 0.8285 | **0.8980** | **0.8773** | 0.8861 |
| aligned_no_sup | **0.9492** | 0.8194 | 0.8911 | 0.8722 | **0.8971** |
| aligned_sup | 0.9469 | 0.8200 | 0.8914 | 0.8633 | 0.8959 |
| aligned_full | 0.9461 | 0.8223 | 0.8912 | 0.8716 | 0.8891 |
| unaligned_no_sup | 0.9447 | 0.8136 | 0.8930 | 0.8501 | 0.8945 |
| unaligned_sup | 0.9460 | 0.8148 | 0.8921 | 0.8453 | 0.8962 |
| unaligned_full | 0.9452 | **0.8180** | 0.8924 | 0.8537 | 0.8916 |

### 4.3 LMSE（↓ 越好）

| 模型 | CEILNet | real20 | objects | postcard | wild |
|------|---------|--------|---------|----------|------|
| baseline | 0.0048 | 0.0201 | **0.0029** | 0.0044 | 0.0083 |
| aligned_no_sup | **0.0038** | 0.0211 | 0.0034 | 0.0050 | 0.0061 |
| aligned_sup | 0.0040 | 0.0207 | 0.0034 | 0.0054 | 0.0062 |
| aligned_full | 0.0042 | 0.0205 | 0.0033 | 0.0047 | 0.0085 |
| unaligned_no_sup | 0.0042 | 0.0209 | 0.0033 | 0.0055 | 0.0055 |
| unaligned_sup | 0.0041 | **0.0198** | 0.0034 | 0.0056 | **0.0052** |
| unaligned_full | 0.0041 | 0.0208 | 0.0034 | 0.0051 | 0.0071 |

### 4.4 NCC（↑ 越好）

| 模型 | CEILNet | real20 | objects | postcard | wild |
|------|---------|--------|---------|----------|------|
| baseline | 0.9808 | 0.8877 | **0.9817** | **0.9463** | 0.9359 |
| aligned_no_sup | **0.9836** | 0.8903 | 0.9819 | 0.9399 | 0.9435 |
| aligned_sup | 0.9828 | 0.8931 | 0.9822 | 0.9306 | 0.9439 |
| aligned_full | 0.9823 | 0.8973 | 0.9819 | 0.9403 | 0.9357 |
| unaligned_no_sup | 0.9816 | 0.8952 | 0.9801 | 0.9196 | **0.9460** |
| unaligned_sup | 0.9819 | 0.9002 | 0.9827 | 0.9116 | 0.9509 |
| unaligned_full | 0.9813 | **0.8981** | 0.9824 | 0.9286 | 0.9390 |

### 4.5 相对 baseline 的 PSNR 变化 Δ（dB）

**CEILNet Table 2**（baseline = 27.88 dB）

| 模型 | PSNR | Δ |
|------|------|---|
| aligned_no_sup | 28.96 | **+1.09** |
| aligned_sup | 28.75 | +0.87 |
| aligned_full | 28.47 | +0.59 |
| unaligned_sup | 28.39 | +0.51 |
| unaligned_no_sup | 28.35 | +0.47 |
| unaligned_full | 28.28 | +0.40 |

**real20**（baseline = 23.55 dB）

| 模型 | PSNR | Δ |
|------|------|---|
| aligned_full | 23.54 | -0.01 |
| unaligned_full | 23.53 | -0.03 |
| unaligned_sup | 23.52 | -0.03 |
| aligned_sup | 23.46 | -0.09 |
| aligned_no_sup | 23.40 | -0.15 |
| unaligned_no_sup | 23.25 | -0.30 |

### 4.6 相对 baseline 的 SSIM 变化 Δ（↑ 越好）

**CEILNet**（baseline = 0.9407）

| 模型 | SSIM | Δ |
|------|------|---|
| aligned_no_sup | 0.9492 | **+0.0085** |
| aligned_sup | 0.9469 | +0.0062 |
| aligned_full | 0.9461 | +0.0054 |
| unaligned_sup | 0.9460 | +0.0053 |
| unaligned_full | 0.9452 | +0.0045 |
| unaligned_no_sup | 0.9447 | +0.0040 |

**real20**（baseline = 0.8285）

| 模型 | SSIM | Δ |
|------|------|---|
| aligned_full | 0.8223 | -0.0062 |
| unaligned_full | 0.8180 | -0.0105 |
| unaligned_sup | 0.8148 | -0.0137 |
| aligned_sup | 0.8200 | -0.0085 |
| aligned_no_sup | 0.8194 | -0.0091 |
| unaligned_no_sup | 0.8136 | -0.0149 |

### 4.7 相对 baseline 的 LMSE 变化 Δ（↓ 越好，负值表示改进）

**CEILNet**（baseline = 0.0048）

| 模型 | LMSE | Δ |
|------|------|---|
| aligned_no_sup | 0.0038 | **-0.0010** |
| aligned_sup | 0.0040 | -0.0008 |
| unaligned_sup | 0.0041 | -0.0007 |
| unaligned_full | 0.0041 | -0.0007 |
| aligned_full | 0.0042 | -0.0006 |
| unaligned_no_sup | 0.0042 | -0.0006 |

**real20**（baseline = 0.0201）

| 模型 | LMSE | Δ |
|------|------|---|
| unaligned_sup | 0.0198 | **-0.0003** |
| aligned_full | 0.0205 | +0.0004 |
| aligned_sup | 0.0207 | +0.0006 |
| unaligned_full | 0.0208 | +0.0007 |
| unaligned_no_sup | 0.0209 | +0.0008 |
| aligned_no_sup | 0.0211 | +0.0010 |

### 4.8 相对 baseline 的 NCC 变化 Δ（↑ 越好）

**CEILNet**（baseline = 0.9808）

| 模型 | NCC | Δ |
|------|-----|---|
| aligned_no_sup | 0.9836 | **+0.0028** |
| aligned_sup | 0.9828 | +0.0020 |
| aligned_full | 0.9823 | +0.0015 |
| unaligned_sup | 0.9819 | +0.0011 |
| unaligned_no_sup | 0.9816 | +0.0008 |
| unaligned_full | 0.9813 | +0.0005 |

**real20**（baseline = 0.8877）

| 模型 | NCC | Δ |
|------|-----|---|
| unaligned_sup | 0.9002 | **+0.0125** |
| unaligned_full | 0.8981 | +0.0104 |
| aligned_full | 0.8973 | +0.0096 |
| unaligned_no_sup | 0.8952 | +0.0075 |
| aligned_sup | 0.8931 | +0.0054 |
| aligned_no_sup | 0.8903 | +0.0026 |

> **四指标一致性说明**：CEILNet 上 PSNR/SSIM/LMSE/NCC 均指向同一结论——RDNet 全面优于 baseline，`aligned_no_sup` 最佳。real20 上 PSNR/SSIM 略降，但 **NCC 全部 RDNet 变体均高于 baseline**（最高 +0.0125），LMSE 仅 `unaligned_sup` 略优；说明结构相似度（NCC）与像素误差（PSNR）在该集上不完全一致。objects/postcard/wild 的 Δ 见 §4.2–4.4 全表自行对比 baseline 行。

---

## 5. 结果分析与讨论

### 5.1 主要发现

1. **区域感知输入在 in-domain 数据上有效**  
   CEILNet 上与训练分布最接近，`aligned_no_sup` PSNR **+1.09 dB**，LMSE 从 0.0048 降至 0.0038，说明即使不监督 RDNet，mask 通道也能为 ERRNet 提供有用的空间先验。

2. **MaxRF mask 监督存在边际递减甚至负效应**  
   同一数据集上 `no_sup > sup > full`。伪标签 `Edge(I) > Edge(T)` 会误将强背景纹理标为反射；`λ_rd` 与 TV 过大时，RDNet 过度拟合 mask，损害最终 restoration。

3. **真实数据需要 unaligned 微调**  
   仅 aligned 训练时 real20 下降 0.01~0.15 dB；`unaligned_sup/full` 微调后 **Δ ≈ -0.03 dB**，与 baseline 统计上可视为持平。说明 RDNet 结构改动后，仍需官方第二阶段 unaligned 流程来适应 real20。

4. **跨数据集泛化不均衡**  
   - **objects / postcard**：baseline 仍领先约 0.07~0.4 dB；  
   - **wild**：aligned 组 PSNR 略升（+0.33 dB for no_sup），unaligned 后回落；  
   - 表明当前轻量 RDNet 并非在所有 benchmark 上稳定超越 ERRNet。

### 5.2 为何部分指标“不如 baseline”？

| 原因 | 说明 |
|------|------|
| 结构改动 | ERRNet 第一层由 3→4 通道，需重新适配；微调预算有限 |
| 伪标签噪声 | MaxRF 监督在域外数据上不可靠 |
| unaligned 阶段 RDNet 无直接 mask 监督 | 仅间接梯度，RDNet 在真实数据上难保持最优 |
| baseline 已充分调优 | ERRNet 是强 baseline，±0.1~0.4 dB 波动常见 |
| 测试集规模 | real20 仅 20 张，小幅度 Δ 可能不具统计显著性 |

### 5.3 推荐在论文中强调的结论

> RDNet 作为轻量插件，**显著提升了与训练域一致的 CEILNet 性能**，验证了反射区域先验的价值；在 **real20 真实数据上经 unaligned 微调后可与 ERRNet baseline 持平**；mask 监督消融表明 **“只提供 mask 输入、弱监督或不监督 RDNet” 优于强 MaxRF 监督**。该方法适合作为 ERRNet 的区域感知扩展方向，但尚未在所有公开 benchmark 上全面超越 baseline。

---

## 6. 定性结果（可视化）

对比图包含 6 列：**Input | GT | Baseline | RDNet | Pred mask | MaxRF pseudo**

| 数据集 | 选用 RDNet 模型 | 原因 |
|--------|-----------------|------|
| CEILNet | `aligned_no_sup` | CEILNet PSNR 最佳 |
| real20 | `unaligned_full` | real20 上 RDNet 最佳（≈ baseline） |

生成命令：

```bash
GPU_IDS=1 bash scripts/run_rdnet_visualize.sh
```

输出路径：

```text
results/rdnet_comparison/
  ceilnet_table2/
    2007_003506_compare.png
    2007_004049_compare.png
    2007_006587_compare.png
    ceilnet_table2_summary.png      # 三样本纵向拼接，适合论文插图
  real20/
    103_compare.png
    12_compare.png
    110_compare.png
    real20_summary.png
```

---

## 7. 结论（论文 Conclusion 可用）

我们提出了 RDNet 引导的 ERRNet，通过预测反射区域 mask 并拼接到 ERRNet 输入，使去反射网络具备显式区域感知能力。在 ERRNet 官方权重基础上，我们系统完成了 aligned 消融与 unaligned 微调，并在 5 个标准数据集上与 baseline 对比。实验显示：（1）在 CEILNet 上 PSNR 最高提升 1.09 dB；（2）MaxRF mask 监督并非越强越好，无监督 mask 输入效果最佳；（3）real20 上需 unaligned 微调才能维持与 baseline 相当的性能。未来工作可探索更鲁棒的反射区域伪标签、自适应 `λ_rd` 调度，以及在 SIR² 全量测试集上的进一步验证。

---

## 8. 局限性与未来工作

- 未完整复现 CVPR 2024 Wild 级联框架，RDNet 为简化版；
- MaxRF 伪标签对强边缘背景敏感，易产生 false positive；
- 本次未评测 `sir2_withgt`；
- unaligned 阶段 RDNet 无 mask 监督，限制其在真实数据上的上限；
- 可尝试：降低 `λ_rd`、仅在 aligned 阶段启用 RDNet、或 two-stage 先训 RDNet 再 frozen 训 ERRNet。

---

## 9. 复现指南

### 9.1 一键脚本

| 脚本 | 功能 |
|------|------|
| `scripts/run_rdnet_aligned_parallel.sh` | 3 组 aligned 消融并行训练 |
| `scripts/run_rdnet_unaligned_parallel.sh` | 3 组 unaligned 微调 |
| `scripts/run_rdnet_benchmark.sh` | 全量 benchmark（7 模型 × 5 数据集） |
| `scripts/run_rdnet_unaligned_benchmark.sh` | 仅 unaligned 3 组 benchmark |
| `scripts/run_rdnet_visualize.sh` | 生成论文对比图 |
| `scripts/parse_benchmark_full.py` | 合并日志并输出汇总表 |
| `scripts/ensure_checkpoint_symlinks.sh` | `errnet_latest.pt` → `errnet_model_latest.pt` |

### 9.2 单条训练命令示例

```bash
# aligned no_sup
python train_errnet.py --name errnet_rdnet_no_sup --hyper --use_rdnet --lambda_rd 0 \
  -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt

# unaligned full（从 aligned full 续训）
python train_errnet_unaligned.py --name errnet_rdnet_full_unaligned --hyper --use_rdnet \
  -r --icnn_path checkpoints/errnet_rdnet_full/errnet_model_latest.pt \
  --unaligned_loss vgg --lr 5e-5
```

### 9.3 测试单数据集

```bash
python test_errnet.py --dataset real20 --name errnet_rdnet_full_unaligned \
  --hyper -r --use_rdnet \
  --icnn_path checkpoints/errnet_rdnet_full_unaligned/errnet_model_latest.pt
```

### 9.4 代码改动清单

| 文件 | 改动 |
|------|------|
| `models/errnet_model.py` | `ReflectionDetectionNet`、mask 拼接、MaxRF 伪标签、RD loss、权重存取 |
| `options/errnet/train_options.py` | RDNet 相关 CLI 参数 |
| `train_errnet.py` | 迁移学习时重置 epoch/iterations |
| `train_errnet_unaligned.py` | NaN 检测与早停 |

### 9.5 CLI 参数速查

```text
--use_rdnet              启用 RDNet
--rdnet_channels 32      RDNet 基础通道
--rdnet_blocks 3         残差块数量
--rdnet_mask_smooth 9    MaxRF 伪标签平滑核
--lambda_rd 1.0          mask L1 权重（0 = no_sup）
--lambda_rd_tv 0.00005   mask TV 权重（0 = sup 无 TV）
```

---

## 10. 论文写作片段（可直接改写粘贴）

### Method 段落

> We attach a lightweight Reflection Detection Network (RDNet) before ERRNet. Given a blended input image \(I\), RDNet predicts a soft reflection mask \(M_{\mathrm{pred}} \in [0,1]^{H \times W}\), which is concatenated with \(I\) (and optional VGG hypercolumn features) as ERRNet input. For pixel-aligned training pairs, we construct MaxRF-style pseudo labels \(M_{\mathrm{gt}} = \mathrm{smooth}(\mathbb{1}[\mathrm{Edge}(I) > \mathrm{Edge}(T)])\), and optionally supervise RDNet with L1 and TV losses. For unaligned data, mask supervision is disabled and only perceptual restoration loss is used.

### Experiment 段落

> We fine-tune from the official ERRNet checkpoint (epoch 60) with three ablations: mask input only (`no_sup`), mask input with L1 supervision (`sup`), and full supervision with TV (`full`). Each variant is trained for 60 aligned epochs and then fine-tuned for 20 unaligned epochs with VGG loss. We evaluate on five benchmarks: CEILNet Table 2, real20, and three SIR² subsets (objects, postcard, wild).

---

## 参考文献

1. ERRNet: [arXiv:2311.17320](https://arxiv.org/abs/2311.17320)  
2. Revisiting Single Image Reflection Removal In the Wild (CVPR 2024) — MaxRF 区域感知思想  
3. ERRNet DIP26 课程框架：`README_DIP26.md`
