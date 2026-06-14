# RIP-ERRNet：基于反射强度先验引导的 ERRNet

论文链接：https://arxiv.org/abs/2312.03798

本分支在 ERRNet baseline 上加入了一个轻量级 Reflection Prior Extraction Network（RPEN）。它不再只判断“哪里有反射”，而是预测每个区域的反射强度先验 reflection-intensity prior，并将该 prior 作为额外通道输入 ERRNet，引导模型进行更有针对性的去反射。

## 方法动机

原始 ERRNet 直接从带反射图像 `I` 恢复干净 transmission 图像 `T`。这种全局恢复方式容易出现两个问题：

```text
1. 局部强反射区域恢复不足。
2. 无反射或弱反射背景区域被过度修改。
```

受 reflection intensity prior 相关工作的启发，本方法显式建模反射强度分布：

```text
反射越强的区域，模型应该更积极地修复。
反射越弱的区域，模型应该尽量保持输入背景结构。
```

因此，本分支引入 RPEN 预测一个连续的强度图，而不是二值 mask：

```text
I -> RPEN -> P_pred
concat(I, P_pred) -> ERRNet -> T_pred
```

其中：

```text
I      : 带反射输入图像
P_pred : 预测的反射强度先验图，范围 [0, 1]
T_pred : ERRNet 输出的去反射结果
```

## 反射强度先验标签

训练时 paired data 中有干净目标图 `T`，所以可以构造伪标签：

```text
P_gt = normalize(mean(|I - T|))
```

含义是：

```text
I 和 T 差异越大，说明该区域反射或退化越强。
I 和 T 差异越小，说明该区域更接近干净背景。
```

代码中还会对 `P_gt` 做平均池化平滑，使其更像区域级 prior，而不是噪声很强的像素级差异图。

## 网络流程

开启 `--use_rpen` 后，训练和推理流程如下：

```text
1. 输入 I
2. P_pred = RPEN(I)
3. I_guided = concat(I, P_pred)
4. T_pred = ERRNet(I_guided)
```

如果同时开启 `--hyper`，VGG hypercolumn 仍然会继续使用：

```text
ERRNet input = concat(I, P_pred, VGG(I))
```

因此，输入通道变化为：

```text
baseline ERRNet: RGB
RIP-ERRNet:      RGB + reflection prior
```

## Loss 设计

本方法保留 ERRNet 原始 restoration loss：

```text
L_restore = L_pixel(T_pred, T) + lambda_vgg * L_vgg(T_pred, T)
```

并新增三个 loss。

### 1. Prior Prediction Loss

监督 RPEN 学会预测反射强度先验：

```text
L_prior = L1(P_pred, P_gt)
```

### 2. Background Preservation Loss

低 prior 区域被认为是弱反射或无反射背景区域。在这些区域中，模型输出不应该偏离输入太多：

```text
L_bg = L1(T_pred * (1 - P_gt), I * (1 - P_gt))
```

这个 loss 用于减少 ERRNet 对干净背景的过度修改。

### 3. Prior TV Smoothness Loss

为了避免 `P_pred` 过于噪声化，加入 TV loss：

```text
L_tv = TV(P_pred)
```

最终总 loss：

```text
L_total = L_restore
        + lambda_prior * L_prior
        + lambda_bg * L_bg
        + lambda_prior_tv * L_tv
```

## 修改的文件

- `models/errnet_model.py`
  - 新增 `ReflectionPriorExtractionNet`。
  - 将 `P_pred` 拼接到 ERRNet 输入。
  - 新增 `P_gt = normalize(mean(|I - T|))` 的构造逻辑。
  - 新增 prior loss、background preservation loss 和 TV loss。
  - 支持保存和加载 RPEN 权重。

- `options/errnet/train_options.py`
  - 新增 RIP-ERRNet 相关命令行参数。

## 新增参数

```text
--use_rpen
    启用 reflection-intensity prior 分支。

--rpen_channels
    RPEN 的基础通道数。默认值：32。

--rpen_blocks
    RPEN 中 residual blocks 的数量。默认值：3。

--rpen_patch_size
    对预测 prior 做 patch-aware 平滑的 average pooling kernel size。
    默认值：15。

--rpen_target_smooth
    对 prior 伪标签 P_gt 做平滑的 average pooling kernel size。
    默认值：15。

--lambda_prior
    prior prediction loss 权重。默认值：1.0。

--lambda_bg
    background preservation loss 权重。默认值：0.5。

--lambda_prior_tv
    predicted prior 的 TV smoothness loss 权重。默认值：0.00005。
```

## 从头训练

```bash
python train_errnet.py \
  --name rip_errnet \
  --hyper \
  --use_rpen \
  --lambda_prior 1.0 \
  --lambda_bg 0.5 \
  --lambda_prior_tv 0.00005
```

## 基于 ERRNet Baseline 微调

可以加载原始 ERRNet 的预训练权重。由于 `--use_rpen` 会让 ERRNet 多接收一个 prior 通道，加载器会自动复制兼容权重，并把新增 prior 通道初始化为 0。

```bash
python train_errnet.py \
  --name rip_errnet_ft \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --lambda_prior 1.0 \
  --lambda_bg 0.5 \
  --lambda_prior_tv 0.00005
```

## Unaligned 数据微调

RIP-ERRNet 也可以接到 unaligned fine-tuning 中：

```bash
 CUDA_VISIBLE_DEVICES=0 nohup python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_full_unaligned \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/rip_full/errnet_latest.pt

 CUDA_VISIBLE_DEVICES=1 nohup python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_input_only_unaligned \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/rip_input_only/errnet_latest.pt

 CUDA_VISIBLE_DEVICES=2 nohup python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_prior_bg_unaligned \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/rip_prior_bg/errnet_latest.pt
  
 CUDA_VISIBLE_DEVICES=3 nohup python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_prior_sup_unaligned \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/rip_prior_sup/errnet_latest.pt
```

注意：对于 unaligned samples，`P_gt = normalize(|I - T|)` 不再可靠，因此 prior supervision 和 background preservation loss 只在 aligned samples 上启用。RPEN 仍然会通过最终 restoration loss 被间接更新。

## 测试

测试时没有 ground truth `T`，所以不会构造 `P_gt`。推理流程只需要：

```text
I -> RPEN -> P_pred -> ERRNet -> T_pred
```

运行命令：

```bash
python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_errnet_ft \
  --hyper \
  --use_rpen \
  -r
```

如果要指定 checkpoint：

```bash
python test_errnet.py \
  --dataset ceilnet_table2 \
  --name rip_errnet_ft \
  --hyper \
  --use_rpen \
  -r \
  --icnn_path checkpoints/rip_errnet_ft/errnet_latest.pt
```

## 推荐消融实验

建议课程项目中做如下消融：

```text
1. ERRNet baseline
2. ERRNet + prior input，不加 prior supervision
3. ERRNet + prior input + prior supervision
4. ERRNet + prior input + prior supervision + background preservation
5. Full RIP-ERRNet：再加入 prior TV smoothness
```

示例命令：

```bash
# 2. 只加入 prior 输入
python train_errnet.py --name rip_input_only --hyper --use_rpen --lambda_prior 0 --lambda_bg 0 --lambda_prior_tv 0

# 3. 加入 prior supervision
CUDA_VISIBLE_DEVICES=1 python train_errnet.py --name rip_prior_sup --hyper --use_rpen --lambda_prior 1.0 --lambda_bg 0 --lambda_prior_tv 0

# 4. 加入 background preservation
CUDA_VISIBLE_DEVICES=2 python train_errnet.py --name rip_prior_bg --hyper --use_rpen --lambda_prior 1.0 --lambda_bg 0.5 --lambda_prior_tv 0

# 5. 完整方法
CUDA_VISIBLE_DEVICES=3 python train_errnet.py --name rip_full --hyper --use_rpen --lambda_prior 1.0 --lambda_bg 0.5 --lambda_prior_tv 0.00005
```

推荐指标：

```text
PSNR / SSIM / LMSE / NCC
```

推荐可视化：

```text
input image I
ground-truth T
baseline ERRNet output
RIP-ERRNet output
P_gt = normalize(mean(|I - T|))
P_pred = RPEN(I)
error map
```

## 可以写进报告的贡献点

```text
1. 提出一个轻量 reflection-intensity prior 分支 RPEN，用于估计反射强度空间分布。
2. 将预测 prior 作为额外通道输入 ERRNet，使去反射网络获得区域强度先验。
3. 设计 background preservation loss，约束低反射区域不要被过度修改。
```

## 注意事项

- `P_gt = normalize(mean(|I - T|))` 是伪标签，不是真实 reflection layer 标注。
- 如果 `lambda_bg` 太大，模型可能变得过度保守，强反射区域去除不充分。
- 如果 `P_pred` 很噪，可以适当增大 `lambda_prior_tv` 或 `rpen_patch_size`。
- 如果输出细节变糊，可以减小 `rpen_target_smooth` 或降低 `lambda_bg`。
