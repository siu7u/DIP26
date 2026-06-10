# RDNet 引导的 ERRNet

本分支在 ERRNet 前面加入了一个轻量级反射区域检测网络 RDNet。核心目标是让去反射模型具备区域感知能力：模型先预测图像中可能存在反射的位置，再把预测得到的反射 mask 作为额外输入通道送入 ERRNet。

## 思路来源

https://arxiv.org/abs/2311.17320

原始 ERRNet 是一个全局图像到图像的恢复网络，输入带反射图像，直接输出去反射后的 transmission/background 图像。这个设计在合成数据和严格对齐数据上比较有效，但真实反射往往具有明显的空间不均匀性：有些区域存在强烈局部反射，而很多背景区域其实不应该被过度修改。

近年的去反射工作，例如 *Revisiting Single Image Reflection Removal In the Wild*（CVPR 2024），强调了反射区域感知的重要性。该工作中的 MaxRF 思路通过比较混合输入图像和干净 transmission 图像的边缘强度来估计局部反射区域。如果输入图像在某个位置的梯度强于干净目标图像，那么这个位置更可能包含局部反射。

本项目没有完整复现 CVPR 2024 的级联框架，而是在 ERRNet baseline 上实现了一个适合课程项目的简化版本：

```text
输入图像 I
  -> RDNet 预测反射区域 mask M_pred
  -> 拼接 [I, M_pred]
  -> ERRNet 输出去反射图像 T_pred
```

训练时，因为 paired data 中有干净目标图像 `T`，所以可以构造一个 MaxRF 风格的伪标签：

```text
M_gt = 1 if Edge(I) > Edge(T)
```

其中 `I` 是带反射输入图像，`T` 是干净 transmission 目标图像。

## 方法说明

这个分支主要包含三个部分。

1. `ReflectionDetectionNet`

   一个小型卷积网络，输入 RGB 图像，输出 1 通道反射区域 mask。

2. Mask 引导的 ERRNet 输入

   开启 `--use_rdnet` 后，模型会把 RDNet 预测出的 mask 和原始输入图像拼接起来。也就是说，ERRNet 的输入从原来的 3 通道变成 4 通道：

   ```text
   baseline: RGB
   RDNet:    RGB + predicted reflection mask
   ```

   如果同时开启 `--hyper`，VGG hypercolumn 特征仍然会继续拼接到后面。

3. 反射区域 mask 监督

   RDNet 使用 MaxRF 风格的伪 mask 进行监督。训练时会加入一个 L1 mask loss，同时加入一个很小的 TV loss，让预测 mask 更平滑，减少噪声。

新增的 RDNet loss 为：

```text
L_rd = lambda_rd * L1(M_pred, M_gt)
     + lambda_rd_tv * TV(M_pred)
```

这个 mask 监督只会用于 aligned training samples。对于 unaligned samples，由于输入图和目标图并不是像素级严格对齐，直接构造 `Edge(I) > Edge(T)` 会不可靠，所以代码会跳过 RD mask supervision。

## 修改的文件

- `models/errnet_model.py`
  - 新增 `ReflectionDetectionNet`。
  - 在 ERRNet 输入前拼接预测 reflection mask。
  - 新增 MaxRF 风格伪 mask 的构造逻辑。
  - 新增 RD mask loss 和 TV smooth loss。
  - 支持保存和加载 RDNet 权重。

- `options/errnet/train_options.py`
  - 新增 RDNet 相关命令行参数。

## 新增参数

```text
--use_rdnet
    启用 RDNet 反射区域检测分支。

--rdnet_channels
    RDNet 的基础通道数。默认值：32。

--rdnet_blocks
    RDNet 中 residual blocks 的数量。默认值：3。

--rdnet_mask_smooth
    对 MaxRF 风格伪 mask 做平滑时使用的 average pooling kernel size。
    默认值：9。

--lambda_rd
    RDNet mask prediction loss 的权重。默认值：1.0。

--lambda_rd_tv
    预测 mask 的 TV smoothness loss 权重。默认值：0.00005。
```

## 从头训练

```bash
python train_errnet.py \
  --name errnet_rdnet \
  --hyper \
  --use_rdnet \
  --lambda_rd 1.0 \
  --lambda_rd_tv 0.00005
```

## 基于 ERRNet Baseline 微调

可以直接使用原始 ERRNet 的预训练权重。开启 `--use_rdnet` 后，ERRNet 会多接收一个 mask 通道，因此第一层卷积的输入通道数会发生变化。代码中的加载逻辑会自动复制兼容的预训练权重，并将新增的 mask 通道初始化为 0。

```bash
python train_errnet.py \
  --name errnet_rdnet_ft \
  --hyper \
  --use_rdnet \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --lambda_rd 1.0 \
  --lambda_rd_tv 0.00005
```

## Unaligned 数据微调

RDNet 也可以在 unaligned fine-tuning 阶段启用：

```bash
python train_errnet_unaligned.py \
  --name errnet_rdnet_unaligned_ft \
  --hyper \
  --use_rdnet \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --unaligned_loss ctx_vgg
```

对于 unaligned samples，RD mask supervision 会被跳过，因为输入和目标之间没有严格像素对齐关系。不过 RDNet 仍然会通过最终的 restoration loss 被更新，因为它预测的 mask 会影响 ERRNet 的输出。

## 测试

测试时使用原来的 `test_errnet.py`，并在加载 RDNet checkpoint 时加上 `--use_rdnet`：

```bash
python test_errnet.py \
  --dataset ceilnet_table2 \
  --name errnet_rdnet_ft \
  --hyper \
  --use_rdnet \
  -r
```

如果需要指定 checkpoint 路径：

```bash
python test_errnet.py \
  --dataset ceilnet_table2 \
  --name errnet_rdnet_ft \
  --hyper \
  --use_rdnet \
  -r \
  --icnn_path checkpoints/errnet_rdnet_ft/errnet_model_latest.pt
```

## 推荐消融实验

课程项目中建议做如下消融：

```text
1. ERRNet baseline
2. ERRNet + RDNet input mask，不加 RD mask supervision
3. ERRNet + RDNet + MaxRF mask supervision
4. ERRNet + RDNet + MaxRF mask supervision + TV mask smoothing
```

示例命令：

```bash
# 2. 只加入 mask 输入，不监督 RDNet
python train_errnet.py --name errnet_rdnet_no_sup --hyper --use_rdnet --lambda_rd 0

# 3. 加入 mask 输入和 MaxRF 伪标签监督
python train_errnet.py --name errnet_rdnet_sup --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0

# 4. 完整方法
python train_errnet.py --name errnet_rdnet_full --hyper --use_rdnet --lambda_rd 1.0 --lambda_rd_tv 0.00005
```

推荐使用的指标：

```text
PSNR / SSIM / LMSE / NCC
```

推荐做的可视化对比：

```text
input image
baseline ERRNet output
RDNet-guided ERRNet output
predicted reflection mask
MaxRF pseudo mask
ground-truth transmission
```

## 可以写进报告的实验动机

可以将本方法描述为：

```text
原始 ERRNet 对整张图像进行全局恢复，容易在无反射区域产生不必要修改，也可能在局部强反射区域优化不足。为此，我们引入一个轻量反射区域检测分支，显式预测 reflection mask，并将其作为空间先验输入 ERRNet，使模型更加关注反射区域，同时尽量保留非反射背景区域。
```

也可以强调这个方法和近年工作的关系：

```text
本方法受到 reflection-region-aware 去反射方法的启发，特别是 CVPR 2024 工作中基于 MaxRF 的局部反射区域估计思想。不同的是，我们没有引入复杂的大规模级联模型，而是在 ERRNet baseline 上设计了一个轻量级、易于消融验证的 RDNet 分支。
```

## 注意事项

- 该方法是一个轻量级课程项目改进，不是对 CVPR 2024 方法的完整复现。
- MaxRF 风格伪 mask 是弱监督标签，可能会把某些强背景边缘误判为反射区域。
- `lambda_rd` 不宜过大，否则模型可能过度关注 mask prediction，而影响最终图像恢复质量。
- 最核心的实验问题是：显式反射区域感知能否帮助 ERRNet 更好地去除局部反射，同时减少对无反射背景区域的破坏。
