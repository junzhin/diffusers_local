# Pipeline 数据输入格式约定

本文档从源码层面精确梳理 diffusers 各主要 text-to-image / text-to-video pipeline 的输入格式要求，内容可直接用于代码修改指导。

**代码路径前缀**：`src/diffusers/pipelines/`

---

## 快速对比表

| Pipeline | 空间约束 | 时序约束 | Latent 通道 | 时序压缩比 | 空间压缩比 | 文本编码器 |
|----------|---------|---------|------------|---------|---------|---------|
| **StableDiffusion** | H,W `% 8 == 0` | — | 4 | — | 8× | CLIP |
| **SDXL** | H,W `% 8 == 0` | — | 4 | — | 8× | CLIP + OpenCLIP |
| **SD3** | H,W `% 16 == 0` | — | 16 | — | 8× | CLIP + OpenCLIP + T5 |
| **Flux** | H,W `% 16 == 0` | — | 16 | — | 8× | CLIP + T5（max 512 token） |
| **Wan** | H,W `% 16 == 0` | `(F-1) % 4 == 0` | 16 | 4× | 8× | UMT5（max 512 token） |
| **Helios** | H,W `% 16 == 0` | `(F-1) % 4 == 0` | 16 | 4× | 8× | UMT5（max 512 token） |
| **LTX** | H,W `% 32 == 0` | `(F-1) % 8 == 0` | 128 | 8× | 32× | T5（max 128 token） |
| **CogVideoX** | H,W `% 16 == 0` | `(F-1) % 4 == 0` | 16 | 4× | 8× | T5（max 226 token） |

---

## 1. 空间维度约束

### 1.1 约束来源

空间约束 = VAE 下采样倍数 × Transformer patch size

```
check_inputs 中的实际检查（各 pipeline 源文件）：

# Wan / Helios / CogVideoX（来自 check_inputs）
if height % 16 != 0 or width % 16 != 0:
    raise ValueError(f"`height` and `width` have to be divisible by 16")

# LTX（来自 check_inputs, ltx/pipeline_ltx.py:380）
if height % 32 != 0 or width % 32 != 0:
    raise ValueError(f"`height` and `width` have to be divisible by 32")

# Flux（来自 check_inputs, flux/pipeline_flux.py:453）
if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
    # vae_scale_factor = 8，所以是 % 16 == 0
    logger.warning(...)   # Flux 是 warning 不是 raise，会自动 resize
```

### 1.2 Wan 的动态 patch 调整（特殊行为）

Wan pipeline 在 `__call__` 中对 height/width 做了额外的动态对齐（`wan/pipeline_wan.py:490`）：

```python
# patch_size 是 Transformer 的属性，不同配置可能不同
h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]  # 通常 8 * 2 = 16
w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]  # 通常 8 * 2 = 16

calc_height = height // h_multiple_of * h_multiple_of  # 向下取整对齐
calc_width  = width  // w_multiple_of * w_multiple_of

if height != calc_height or width != calc_width:
    # 自动截断到最近的合法值，不抛出异常
    height, width = calc_height, calc_width
```

> **实用结论**：传入任意 height/width 给 Wan 不会报错，但会被静默截断。需要精确控制时，自行预对齐到 16 的倍数。

---

## 2. 时序维度约束（视频 Pipeline 专属）

### 2.1 约束规律

所有视频 pipeline 的时序约束来自于 **VAE 时序压缩比**，且遵循统一模式：

```
合法的 num_frames 满足：(num_frames - 1) % temporal_stride == 0
```

其中第一帧被单独处理（不参与时序压缩），所以需减去 1 再取模。

```python
# Wan（wan/pipeline_wan.py:490）
if num_frames % self.vae_scale_factor_temporal != 1:
    # 自动修正到最近合法值
    num_frames = num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1

# LTX 的 prepare_latents 中实际压缩计算（ltx/pipeline_ltx.py:497）
num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1  # (F-1)//8 + 1
```

### 2.2 各 Pipeline 的合法帧数序列

```python
# Wan（temporal stride = 4）
# 合法值: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81
# 默认: 81 帧

# LTX（temporal stride = 8）
# 合法值: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161
# 默认: 161 帧

# CogVideoX（temporal stride = 4）
# 合法值: 1, 5, 9, 13, ..., 49
# 默认: 49 帧（约 6 秒 @ 8fps）
```

---

## 3. Latent 张量格式

### 3.1 Text-to-Image Pipeline 的 Latent Shape

```python
# SD / SDXL（标准 KL-VAE）
# latent_channels = 4, vae_scale_factor = 8
shape = (batch_size, 4, height // 8, width // 8)

# SD3 / Flux（新一代 KL-VAE）
# latent_channels = 16, vae_scale_factor = 8
shape = (batch_size, 16, height // 8, width // 8)

# 注意：Flux 的 latent 在输入 Transformer 前会 pack 成 sequence 格式
# packed_shape = (batch_size, (H//16 * W//16), 16 * 4)  # patchify 2×2
# 只有 output_type="latent" 时才能获得 packed 格式，需要 _unpack_latents 还原
```

### 3.2 Text-to-Video Pipeline 的 Latent Shape

```python
# Wan（wan/pipeline_wan.py:325）
# latent_channels = 16, spatial_factor = 8, temporal_factor = 4
shape = (batch_size, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
# 例：batch=1, F=81, H=480, W=832 → (1, 16, 21, 60, 104)

# Helios
# latent_channels = 16, spatial_factor = 8（同 Wan），temporal_factor = 4
shape = (batch_size, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)

# LTX（ltx/pipeline_ltx.py:490，DC-AE 架构）
# latent_channels = 128, spatial_factor = 32, temporal_factor = 8
shape = (batch_size, 128, (num_frames - 1) // 8 + 1, height // 32, width // 32)
# 例：batch=1, F=161, H=512, W=704 → (1, 128, 21, 16, 22)
```

### 3.3 传入预计算 Latent（`latents` 参数）

所有视频 pipeline 均支持直接传入已编码的 latent，跳过 VAE encode：

```python
# prepare_latents 中的处理（所有 pipeline 共同模式）
if latents is not None:
    return latents.to(device=device, dtype=dtype)  # 直接使用，不做任何缩放
```

> **注意**：预计算 latent 应已经过 `scaling_factor` 归一化，即 `z = vae.encode(x).latent_dist.sample() * scaling_factor`。

---

## 4. 文本输入格式

### 4.1 Prompt 基本类型

所有 pipeline 的 `prompt` 参数均支持：

```python
prompt: str           # 单个提示词
prompt: list[str]     # 批量，长度必须等于 batch_size
```

### 4.2 各 Pipeline 的文本编码器及 max_sequence_length

| Pipeline | 编码器 | Token 上限 | 特殊说明 |
|----------|--------|-----------|---------|
| **SD** | `CLIPTextModel` | 77 | 超出直接截断 |
| **SDXL** | CLIP + OpenCLIP | 77 × 2 | `prompt_2` 给第二编码器 |
| **SD3** | CLIP + OpenCLIP + T5 | 77 / 77 / 256 | `prompt_2`, `prompt_3` 分别对应 |
| **Flux** | CLIP + T5 | 77 / 512 | `prompt_2` 给 T5；`max_sequence_length ≤ 512`（check_inputs 强制验证）|
| **Wan** | UMT5-XXL | 512 | 单编码器，`max_sequence_length=512` |
| **LTX** | T5-XXL | 128 | **严格限制 128**（`check_inputs` 中 raise）|
| **Helios** | UMT5-XXL | 512 | 单编码器 |
| **CogVideoX** | T5-XXL | 226 | encode_prompt 默认 `max_sequence_length=226` |

### 4.3 双/三编码器 Pipeline 的 prompt 分配

```python
# SDXL / Flux / SD3 支持分别传入两个 prompt
pipe(
    prompt="short style prompt",   # → CLIP 编码器（77 token 限制）
    prompt_2="long detailed description...",  # → T5/OpenCLIP 编码器（更长 token 限制）
)
# 若只传 prompt，框架内部自动复制给所有编码器
```

### 4.4 预嵌入（bypass 编码器）

```python
# 所有 pipeline 支持直接传入预计算的文本嵌入
pipe(
    prompt_embeds=my_embeds,        # shape 依编码器而定
    negative_prompt_embeds=my_neg_embeds,
)

# SD/SDXL 的 pooled embedding（用于调节时间步嵌入）
pipe(
    pooled_prompt_embeds=pooled,
    negative_pooled_prompt_embeds=neg_pooled,
)
```

---

## 5. 图像/视频条件输入（Conditional Pipeline）

### 5.1 Image-to-Video（Helios）

Helios pipeline 支持可选的图像条件化输入（`helios/pipeline_helios.py`）：

```python
pipe(
    image=pil_image,         # PipelineImageInput，用于 i2v 推理
    image_latents=...,       # torch.Tensor，预编码图像 latent（可选）
    fake_image_latents=...,  # torch.Tensor，用于无图像条件时的占位（可选）
)
```

内部处理流程：

```python
# helios/pipeline_helios.py:616
if image is not None:
    image = self.video_processor.preprocess(image, height=height, width=width)
    # preprocess 输出: (B, C, H, W)，值域 [-1, 1]
    image_latents, fake_image_latents = self.prepare_image_latents(
        image,
        latents_mean=latents_mean,   # 来自 vae.config
        latents_std=latents_std,     # 来自 vae.config
        ...
    )
```

### 5.2 VideoProcessor 的标准化约定

`video_processor.preprocess` 对图像/视频做统一预处理，输出格式：

```python
# 单图像
output: torch.Tensor  # shape: (1, C, H, W)，dtype: float32，值域: [-1, 1]

# 视频帧序列
output: torch.Tensor  # shape: (1, C, T, H, W)，dtype: float32，值域: [-1, 1]
```

---

## 6. 各 Pipeline 默认分辨率参考

| Pipeline | 默认 H | 默认 W | 默认帧数 | 等效时长 |
|----------|-------|-------|--------|--------|
| SD v1.5 | 512 | 512 | — | — |
| SDXL | 1024 | 1024 | — | — |
| Flux | 1024 | 1024 | — | — |
| Wan | 480 | 832 | 81 | ~5s @ 16fps |
| Helios | 384 | 640 | 132 | `(132-1)//4+1 = 34` latent frames |
| LTX | 512 | 704 | 161 | ~6.4s @ 25fps |
| CogVideoX | 480 | 720 | 49 | ~6s @ 8fps |

---

## 7. Scaling Factor 与 Latent 归一化约定

```python
# 编码阶段（统一约定）
latent = vae.encode(pixel_values).latent_dist.sample()
latent_scaled = latent * vae.config.scaling_factor   # 存储/传输用

# 解码阶段（统一约定）
image = vae.decode(latent_scaled / vae.config.scaling_factor).sample

# SD3 / Flux 有额外的 shift_factor（vae.config.shift_factor）
image = vae.decode(latent / scaling_factor + shift_factor).sample
```

各模型的参考值：

| 模型 | `scaling_factor` | `shift_factor` |
|------|-----------------|----------------|
| SD 1.x | 0.18215 | — |
| SDXL | 0.13025 | — |
| SD3 | 1.5305 | 0.0609 |
| Wan | 需校准 | — |
| LTX | 需校准 | — |

---

## 8. 代码修改检查清单

修改或新增 pipeline 时需验证的输入格式要素：

```python
# 检查清单
assert height % h_divisor == 0, f"Height must be divisible by {h_divisor}"
assert width % w_divisor == 0,  f"Width must be divisible by {w_divisor}"
assert (num_frames - 1) % temporal_stride == 0, "num_frames constraint violated"
assert len(prompt_embeds.shape) == 3, "prompt_embeds: (B, seq_len, hidden_dim)"
assert latents.shape[1] == latent_channels, "Latent channel mismatch with VAE"
assert latents.shape[-2] == height // spatial_factor
assert latents.shape[-1] == width  // spatial_factor
```

快速验证脚本：

```python
def check_video_input_format(H, W, F, pipeline_type="wan"):
    """验证输入尺寸是否符合指定 pipeline 的要求"""
    constraints = {
        "wan":     {"spatial": 16, "temporal": 4,  "latent_c": 16,  "sp_factor": 8,  "t_factor": 4},
        "ltx":     {"spatial": 32, "temporal": 8,  "latent_c": 128, "sp_factor": 32, "t_factor": 8},
        "helios":  {"spatial": 16, "temporal": 4,  "latent_c": 16,  "sp_factor": 8,  "t_factor": 4},
        "cogvideo":{"spatial": 16, "temporal": 4,  "latent_c": 16,  "sp_factor": 8,  "t_factor": 4},
    }
    c = constraints[pipeline_type]
    ok = True
    if H % c["spatial"] != 0:
        print(f"✗ Height {H} not divisible by {c['spatial']}")
        ok = False
    if W % c["spatial"] != 0:
        print(f"✗ Width {W} not divisible by {c['spatial']}")
        ok = False
    if c["temporal"] and (F - 1) % c["temporal"] != 0:
        print(f"✗ num_frames {F}: (F-1) must be divisible by {c['temporal']}")
        ok = False

    if ok:
        t_factor = c["t_factor"] or 1
        latent_shape = (1, c["latent_c"],
                        (F - 1) // t_factor + 1 if c["t_factor"] else 1,
                        H // c["sp_factor"],
                        W // c["sp_factor"])
        print(f"✓ Valid input for {pipeline_type}")
        print(f"  Latent shape: {latent_shape}")

# 使用示例
check_video_input_format(480, 832, 81, "wan")
# → ✓ Valid input for wan
# → Latent shape: (1, 16, 21, 60, 104)
```
