# VAE 灵活性实验指南

本文档针对两个核心实验需求，从框架源码层面给出完整的技术分析和可执行方案。

**代码均基于** `src/diffusers/` 中的实际实现，关键路径附源文件行号。

---

## 快速决策树

```
你的需求是？
│
├── 推理时不走 Decoder → 1.1 output_type="latent"
├── 推理时传入已编码的 Latent → 1.2 latents= 参数
├── 完全不用 VAE，只操作扩散模型 → 1.3 手写 denoising loop
│
└── 替换 VAE 做实验
    ├── 只想推理对比 → 2.3 方案 A（整体替换 pipe.vae）
    ├── 训练对比（KL vs VQ，VAE 冻结）→ 2.4 方案 B（统一接口封装）
    ├── 只换内部 encoder 或 decoder → 2.5 方案 C（属性替换）
    └── 完全自定义（FSQ/LFQ 等）→ 2.6 方案 D（实现标准接口）
```

---

## 需求一：不经过 Encoder/Decoder 的推理

### 1.1 跳过 Decoder：`output_type="latent"`

所有主流 pipeline 原生支持，推理结束后直接返回 latent tensor，不调用 `vae.decode()`。

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("sd-v1-5", torch_dtype=torch.float16)
pipe.to("cuda")

result = pipe("a photo of a cat", output_type="latent")
latents = result.images  # shape: (1, 4, 64, 64)，已经过 denoising，未解码
```

**框架内部** (`pipeline_stable_diffusion.py:1084`)：

```python
if not output_type == "latent":
    image = self.vae.decode(latents / self.vae.config.scaling_factor, ...)[0]
else:
    image = latents  # 直接返回，完全跳过 vae.decode()
```

不同 pipeline 的 latent 尺寸和特殊处理：

| Pipeline | Latent Shape | 解码前缩放公式 | 特殊处理 |
|----------|-------------|--------------|---------|
| SD 1.x/2.x | `(B, 4, H/8, W/8)` | `z / scaling_factor` | 无 |
| SDXL | `(B, 4, H/8, W/8)` | `z / scaling_factor` | 无 |
| SD3 | `(B, 16, H/8, W/8)` | `z / scaling_factor + shift_factor` | 有 `shift_factor` |
| Flux | `(B, 16, H/8, W/8)` | `z / scaling_factor + shift_factor` | `output_type="latent"` 返回 packed 格式，需 unpack 才能得到空间 tensor |

> **Flux 的 packed 格式**：返回的 latent 是 sequence 维度的 packed 表示。若需要空间格式：
> ```python
> spatial = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
> ```

---

### 1.2 跳过 Encoder：传入预计算 Latent

`prepare_latents` 接受可选的 `latents` 参数。若传入，完全跳过 `vae.encode()`，直接进入 denoising loop。

**SD 的处理** (`pipeline_stable_diffusion.py:694`)：

```python
def prepare_latents(self, ..., latents=None):
    if latents is None:
        latents = randn_tensor(shape, ...)
    else:
        latents = latents.to(device)              # 直接使用，不编码
    latents = latents * self.scheduler.init_noise_sigma  # 调度器缩放
    return latents
```

**SD3 的处理** (`pipeline_stable_diffusion_3.py:633`)：

```python
def prepare_latents(self, ..., latents=None):
    if latents is not None:
        return latents.to(device=device, dtype=dtype)   # 直接返回，不做任何缩放
    # ... 生成随机噪声 ...
```

> 注意两者行为差异：SD 会对传入 latent 乘以 `init_noise_sigma`（对 DDIM 等调度器该值通常为 1），SD3 不做任何处理。建议传入前确认调度器的 `init_noise_sigma` 是否为 1。

**使用示例**：

```python
# 假设已有预编码的 latent（已乘以 scaling_factor）
precomputed = torch.load("my_latent.pt")   # shape: (1, 4, 64, 64)

# 传入后 pipeline 跳过 vae.encode()，直接 denoising
result = pipe("a photo", latents=precomputed, output_type="latent")
```

---

### 1.3 完全绕过 VAE：手写 Denoising Loop

最大控制粒度，仅加载 UNet + Scheduler + Text Encoder，不加载 VAE：

```python
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch

unet = UNet2DConditionModel.from_pretrained("sd-v1-5", subfolder="unet")
scheduler = DDIMScheduler.from_pretrained("sd-v1-5", subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained("sd-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("sd-v1-5", subfolder="text_encoder")

unet.to("cuda")
text_encoder.to("cuda")

# 文本编码
tokens = tokenizer(["a cat"], return_tensors="pt", padding=True).to("cuda")
text_embeds = text_encoder(**tokens).last_hidden_state

# 初始化随机 latent
latents = torch.randn(1, 4, 64, 64, device="cuda")

# Denoising loop（完全没有 VAE）
scheduler.set_timesteps(50)
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = unet(latents, t, encoder_hidden_states=text_embeds).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# latents 即为最终结果，shape: (1, 4, 64, 64)
# 可选：用任意 VAE 解码，或存为 latent 做后续分析
```

---

## 需求二：替换 Encoder/Decoder（VE vs VQVAE 对比实验）

### 2.1 框架内现有 VAE 类型及 encode 输出接口

这是框架层面最需要注意的差异。各 VAE 的 `encode()` **返回类型不同**：

| 类 | encode 返回类型 | 获取 latent 的方式 | decode 是否有 aux loss |
|----|----------------|------------------|----------------------|
| `AutoencoderKL` | `AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)` | `.latent_dist.sample()` 或 `.latent_dist.mode()` | 无 |
| `VQModel` | `VQEncoderOutput(latents=Tensor)` | `.latents` 直接取 | 有 `commit_loss`（decode 时产生） |
| `AutoencoderTiny` | `AutoencoderTinyOutput(latents=Tensor)` | `.latents` 直接取 | 无 |
| `AutoencoderDC` | `EncoderOutput(latent=Tensor)` | `.latent` 直接取 | 无 |

---

### 2.2 接口统一方案：`IdentityDistribution`

框架在 `src/diffusers/models/autoencoders/vae.py:743` 已内置 `IdentityDistribution`，专门用于将确定性 tensor 适配成分布接口：

```python
class IdentityDistribution(object):
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters

    def sample(self, generator=None) -> torch.Tensor:
        return self.parameters   # 直接返回 tensor，无随机性

    def mode(self) -> torch.Tensor:
        return self.parameters   # 同上
```

利用它，可以将 VQModel 的输出无缝包装成 `DiagonalGaussianDistribution` 的接口形式，使所有调用 `.latent_dist.sample()` 的 pipeline 代码零改动：

```python
from diffusers.models.autoencoders.vae import IdentityDistribution

# VQModel 路径
vq_out = vq_vae.encode(pixel_values)        # VQEncoderOutput(latents=Tensor)
latent_dist = IdentityDistribution(vq_out.latents)

# 现在和 KL-VAE 接口完全一致
latent = latent_dist.sample()               # 等价于直接取 .latents
latent = latent_dist.mode()                 # 同上
```

`DiagonalGaussianDistribution` 也有类似机制：`deterministic=True` 时 std=0，`sample()` 等价于 `mode()`，可用于包装任意确定性 encoder：

```python
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# 若 encoder 输出单通道 tensor（非 mean+logvar 拼接），需要手动扩展
fake_params = torch.cat([latent, torch.zeros_like(latent)], dim=1)  # 2*C 通道
dist = DiagonalGaussianDistribution(fake_params, deterministic=True)
# dist.sample() = dist.mode() = latent
```

---

### 2.3 方案 A：整体替换 `pipe.vae`（推理验证首选）

```python
from diffusers import StableDiffusionPipeline, VQModel

pipe = StableDiffusionPipeline.from_pretrained("sd-v1-5")

# 直接替换，VQModel 实现了兼容的 .encode() / .decode() 接口
pipe.vae = VQModel.from_pretrained("your-vqvae-checkpoint")

result = pipe("a photo", output_type="pil")
```

**兼容性检查清单**：

| 项目 | 要求 | 检查方式 |
|------|------|---------|
| `latent_channels` | 必须等于 UNet 的 `in_channels` | `vae.config.latent_channels == unet.config.in_channels` |
| `scaling_factor` | 需重新校准（见 2.8 节） | 统计 latent 标准差 |
| `shift_factor` | SD3/Flux 需要此属性 | `hasattr(vae.config, 'shift_factor')` |
| 空间下采样倍数 | VAE 的缩放比必须与 pipeline 的 `vae_scale_factor` 一致 | 通常为 8 |

**已知问题**：VQModel 的 `encode()` 在 pipeline 内部被调用时（如 img2img），pipeline 会尝试 `.latent_dist.sample()`，而 VQModel 的 encode 返回的是 `.latents`，会报 `AttributeError`。此时需要方案 D 的自定义封装。

---

### 2.4 方案 B：训练脚本条件化替换（对比训练推荐）

参考 `examples/dreambooth/train_dreambooth.py:937` 的 `vae=None` 设计，扩展为支持多种 VAE。

#### 统一 encode 接口封装

```python
from diffusers import AutoencoderKL, VQModel
from diffusers.models.autoencoders.vae import IdentityDistribution
import torch

def get_latent_dist(vae, pixel_values):
    """
    统一各类 VAE 的 encode 接口，始终返回支持 .sample() / .mode() 的分布对象。
    此函数在 torch.no_grad() 外调用（调用方决定是否需要梯度）。
    """
    if vae is None:
        return None  # pixel space 训练

    if isinstance(vae, AutoencoderKL):
        # 返回 DiagonalGaussianDistribution，支持 .sample() / .mode() / .kl()
        return vae.encode(pixel_values).latent_dist

    if isinstance(vae, VQModel):
        # 用 IdentityDistribution 包装，使接口对齐
        latents = vae.encode(pixel_values).latents
        return IdentityDistribution(latents)

    raise ValueError(f"Unsupported VAE type: {type(vae)}")
```

#### 训练循环

```python
# 初始化（选择 VAE 类型）
vae_type = "kl"   # 可切换为 "vq"

if vae_type == "kl":
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
elif vae_type == "vq":
    vae = VQModel.from_pretrained(args.vqvae_path)
else:
    vae = None  # 在 pixel space 训练

# 标准实践：VAE 冻结，只训练 UNet
if vae is not None:
    vae.requires_grad_(False)
    vae.eval()
    # 冻结的 VAE 可以用半精度节省显存
    vae.to(accelerator.device, dtype=weight_dtype)

# 训练 loop
for batch in dataloader:
    with accelerator.accumulate(unet):
        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

        # 编码（VAE 冻结时不需要梯度）
        with torch.no_grad():
            if vae is not None:
                latent_dist = get_latent_dist(vae, pixel_values)
                # 训练时用 sample()（引入随机性）
                model_input = latent_dist.sample() * vae.config.scaling_factor
            else:
                model_input = pixel_values  # pixel space

        # 标准 diffusion loss
        noise = torch.randn_like(model_input)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (model_input.shape[0],), device=model_input.device)
        noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
```

> **注意**：`with torch.no_grad()` 在 VAE 冻结（`requires_grad_(False)`）时是冗余保险，但建议保留以节省显存。若未来需要解冻 VAE 联合训练，需去掉此 context。

---

### 2.5 方案 C：只替换内部 Encoder 或 Decoder 模块

`AutoencoderKL` 的 `self.encoder`、`self.decoder`、`self.quant_conv`、`self.post_quant_conv` 都是独立的 `nn.Module`，支持直接赋值替换。

```python
from diffusers import AutoencoderKL
import torch.nn as nn

vae = AutoencoderKL.from_pretrained("sd-v1-5", subfolder="vae")

# 替换 Encoder（保留 Decoder 和 quant_conv 不变）
class MyEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 你的实现

    def forward(self, x):
        # 输出 shape 必须为 (B, 2*latent_channels, H/f, W/f)
        # 原因：quant_conv 期望的输入是 mean+logvar 的拼接（2*C 通道）
        return encoded

vae.encoder = MyEncoder(...)

# 若你的 encoder 直接输出 mean+logvar（不需要 quant_conv 做线性投影），
# 可将 quant_conv 设为恒等变换以透传
vae.quant_conv = nn.Identity()
vae.post_quant_conv = nn.Identity()
```

**Encoder 输出通道约定**：

原始 `Encoder`（`vae.py`）构造时传入 `double_z=True`，最后一层卷积输出 `2 * latent_channels` 通道，前半为 mean，后半为 logvar，这是 `DiagonalGaussianDistribution` 的输入格式：

```python
# vae.py:690
self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
```

若你的自定义 encoder 只输出 `latent_channels` 通道（确定性，无 logvar），有两个选项：
1. 将 encoder 输出 `cat([h, torch.zeros_like(h)], dim=1)` 手动补零
2. 使用 `IdentityDistribution` 包装，不走 `DiagonalGaussianDistribution`（需改 `encode()` 方法）

---

### 2.6 方案 D：完整自定义 VAE 类

实现与所有 pipeline 完全兼容的自定义 VAE，只需遵守以下接口约定：

```python
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, IdentityDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
import torch, torch.nn as nn


class MyCustomVAE(ModelMixin, AutoencoderMixin, ConfigMixin):
    """
    自定义 VAE，实现与所有 diffusers pipeline 兼容的接口。
    继承 ModelMixin → 自动支持 from_pretrained / save_pretrained。
    继承 AutoencoderMixin → 自动获得 enable_tiling / enable_slicing 方法。
    继承 ConfigMixin → __init__ 参数自动序列化到 config.json。
    """

    @register_to_config           # 必须：__init__ 参数自动写入 config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,   # SD3/Flux 需要
        # 你的自定义参数
    ):
        super().__init__()
        self.encoder = ...   # nn.Module
        self.decoder = ...   # nn.Module
        # AutoencoderMixin 要求以下属性存在
        self.use_slicing = False
        self.use_tiling = False

    @apply_forward_hook           # 必须：使 encode() 与 CPU offloading 兼容
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)

        # 选项 1：返回 KL 分布（适合有 mean+logvar 输出的 encoder）
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(h)   # h 需要是 2*latent_channels 通道
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

        # 选项 2：返回确定性 latent（适合 VQ encoder）
        # posterior = IdentityDistribution(h)          # h 是 latent_channels 通道
        # return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook           # 必须：使 decode() 与 CPU offloading 兼容
    def decode(self, z: torch.Tensor, return_dict: bool = True, **kwargs):
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
```

**Pipeline 实际调用的最小接口**（可用于接口验证测试）：

```python
# img2img / inpaint pipeline 会调用 encode：
latent = vae.encode(pixel_values).latent_dist.sample()
latent = latent * vae.config.scaling_factor

# 所有 pipeline 调用 decode：
image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

# SD3 / Flux 额外使用 shift_factor：
image = vae.decode(
    latents / vae.config.scaling_factor + vae.config.shift_factor,
    return_dict=False
)[0]

# pipeline 也会直接访问：
vae.config.scaling_factor   # float，必须存在
vae.config.shift_factor     # float | None
vae.config.latent_channels  # int（部分 pipeline 检查）
```

---

### 2.7 关于 `AutoencoderKL.forward()` 的特殊用途

`AutoencoderKL.forward(sample, sample_posterior=False)` 是 encode + decode 的组合，主要用于**单独训练 VAE**（非 LDM 训练）时的重建 loss 计算：

```python
# 用于 VAE 重建训练
reconstructed = vae(pixel_values, sample_posterior=True).sample  # 用采样
# 或
reconstructed = vae(pixel_values, sample_posterior=False).sample  # 用 mean（确定性）

# 重建 loss + KL 正则
recon_loss = F.l1_loss(reconstructed, pixel_values)
kl_loss = vae.encode(pixel_values).latent_dist.kl().mean()
total_loss = recon_loss + kl_weight * kl_loss
```

在 LDM 训练中（VAE 冻结），通常不调用 `forward()`，而是分别调用 `encode()` 和 `decode()`。

---

### 2.8 Scaling Factor 校准

VAE 的 latent 分布方差决定了 `scaling_factor`。不同 VAE 的方差差异很大，不校准会导致 UNet 看到的信噪比完全错误，显著影响训练效果。

**校准方法**：

```python
import torch
from tqdm import tqdm

vae.eval()
all_stds = []

with torch.no_grad():
    for batch in calibration_dataloader:   # 推荐 1000 张以上
        if isinstance(vae, AutoencoderKL):
            latents = vae.encode(batch["pixel_values"].to(vae.device)).latent_dist.sample()
        elif isinstance(vae, VQModel):
            latents = vae.encode(batch["pixel_values"].to(vae.device)).latents
        all_stds.append(latents.std().item())

std_mean = torch.tensor(all_stds).mean()
scaling_factor = 1.0 / std_mean
print(f"std_mean={std_mean:.4f}, scaling_factor={scaling_factor:.5f}")

# 写入配置（不持久化，仅本次 session 生效）
vae.config.scaling_factor = scaling_factor.item()
# 持久化：保存到 config.json
vae.save_pretrained("my_vae_calibrated/")
```

**常见参考值**：

| VAE | scaling_factor | 来源 |
|-----|----------------|------|
| SD 1.x KL-VAE | 0.18215 | LDM 论文统计值 |
| SDXL KL-VAE | 0.13025 | 高分辨率调整 |
| SD3 KL-VAE | 1.5305 (+ shift 0.0609) | 带 shift_factor |
| 自定义 VQ-VAE | 需校准 | 取决于 codebook 维度 |

---

### 2.9 联合训练 VAE + UNet 时的 commit_loss 处理

若实验需要**端到端训练**（VAE 参与反向传播），VQModel 的 `commit_loss` 必须加入总 loss。

```python
# 解冻 VAE
vae.requires_grad_(True)
vae.train()

# 训练 loop（注意：不能用 torch.no_grad()）
latents = vae.encode(pixel_values).latents  # VQModel
latents_scaled = latents * vae.config.scaling_factor

# ... diffusion loss 计算 ...
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())

# 获取 commit_loss（decode 时产生）
decode_out = vae.decode(latents)
commit_loss = decode_out.commit_loss.mean()

# commit_loss 权重通常较小（0.1~1.0）
total_loss = diffusion_loss + args.vq_commit_weight * commit_loss
accelerator.backward(total_loss)
```

**`commit_loss` 的来源**（`vae.py:VectorQuantizer.forward()`）：

```
loss = beta * ||z_q.detach() - z||² + ||z_q - z.detach()||²
       ↑ 推动 codebook 向 encoder 靠近    ↑ 推动 encoder 向 codebook 靠近
```
`beta` 默认 0.25，权衡两项优化目标。

---

### 2.10 混合精度训练时的 VAE 处理

VAE 冻结时通常转为半精度节省显存：

```python
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# VAE 冻结 → 转为 weight_dtype（默认 bf16/fp16）
if vae is not None:
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)

# 编码时同步 dtype
model_input = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
```

> **注意**：`AutoencoderKL` 有 `force_upcast=True` 参数，在高分辨率图像下会自动临时提升到 float32 进行编码，避免 fp16 溢出（SDXL VAE 常用）。

---

## 各方案对比总结

| 方案 | 适用场景 | 改动量 | img2img 支持 | 训练支持 | VQ/KL 通用 |
|------|---------|--------|------------|---------|-----------|
| A. `pipe.vae = ...` | 推理对比 | 极小 | 需接口兼容 | 有限 | 需接口兼容 |
| B. 统一接口封装 | 对比训练（VAE 冻结） | 中 | ✅ | ✅ | ✅ |
| C. 属性替换 `vae.encoder` | 只换内部实现 | 小 | ✅ | ✅ | 部分 |
| D. 自定义 VAE 类 | 完全定制 | 较大 | ✅ | ✅ | ✅ |

---

## 快速验证脚本

在实际实验前，用此脚本验证你的 VAE 实现是否满足接口要求：

```python
"""
运行：python docs_experiments/verify_vae_interface.py
验证自定义 VAE 是否满足 diffusers pipeline 的调用要求
"""
import torch
from diffusers import AutoencoderKL, VQModel
from diffusers.models.autoencoders.vae import IdentityDistribution

def verify_vae_interface(vae, latent_channels=4, device="cpu"):
    dummy_image = torch.randn(2, 3, 256, 256).to(device)
    vae = vae.to(device).eval()

    with torch.no_grad():
        # 1. 验证 encode 接口
        enc_out = vae.encode(dummy_image)
        print(f"[encode] output type: {type(enc_out)}")

        # 2. 验证 latent_dist（KL-VAE 路径）
        if hasattr(enc_out, 'latent_dist'):
            dist = enc_out.latent_dist
            latent = dist.sample()
            print(f"[latent_dist.sample] shape: {latent.shape}, mean: {latent.mean():.4f}, std: {latent.std():.4f}")
            latent = dist.mode()
            print(f"[latent_dist.mode]   shape: {latent.shape}")

        # 3. 验证 latents（VQ 路径）
        elif hasattr(enc_out, 'latents'):
            latent = enc_out.latents
            print(f"[latents] shape: {latent.shape}, mean: {latent.mean():.4f}, std: {latent.std():.4f}")
            # 包装为分布接口
            dist = IdentityDistribution(latent)
            print(f"[IdentityDistribution.sample] shape: {dist.sample().shape}")

        # 4. 验证 decode 接口
        dec_out = vae.decode(latent, return_dict=False)
        print(f"[decode] output shape: {dec_out[0].shape}")

        # 5. 验证 config 属性
        print(f"[config] scaling_factor: {vae.config.scaling_factor}")
        print(f"[config] shift_factor: {getattr(vae.config, 'shift_factor', 'N/A')}")
        print(f"[config] latent_channels: {getattr(vae.config, 'latent_channels', 'N/A')}")

    print("✓ Interface verification passed\n")


# 测试 KL-VAE
print("=== AutoencoderKL ===")
kl_vae = AutoencoderKL(
    in_channels=3, out_channels=3, latent_channels=4,
    block_out_channels=(64, 128), layers_per_block=1,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
)
verify_vae_interface(kl_vae)

# 测试 VQModel
print("=== VQModel ===")
vq_vae = VQModel(
    in_channels=3, out_channels=3, latent_channels=4,
    block_out_channels=(64, 128), layers_per_block=1,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    num_vq_embeddings=256,
)
verify_vae_interface(vq_vae)
```
