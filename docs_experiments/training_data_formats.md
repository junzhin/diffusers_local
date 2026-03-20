# 训练数据格式与 DataLoader 配置指南

本文档覆盖 `examples/` 中各类训练脚本的数据格式要求、数据集配置方式、DataLoader 关键参数及其源码位置。

**代码路径前缀**：`examples/`

---

## 1. 两条数据路径概览

diffusers 训练脚本支持两种完全不同的数据读取方式：

| 路径 | 适用脚本 | 数据量 | 格式 | 入口 |
|------|---------|--------|------|------|
| **HuggingFace `datasets` 库** | text_to_image 系列 | 大规模（千~万张） | Hub dataset 或 本地 ImageFolder | `datasets.load_dataset()` |
| **纯文件夹遍历** | dreambooth 系列 | 小规模（3~100 张） | 图像文件夹，无 metadata | `Path(dir).iterdir()` |

---

## 2. text_to_image 系列：数据格式与 DataLoader

### 2.1 支持的数据源

**脚本文件**：`text_to_image/train_text_to_image.py`

```python
# 来源 A：HuggingFace Hub
--dataset_name lambdalabs/naruto-blip-captions
--dataset_config_name None

# 来源 B：本地文件夹（ImageFolder 格式）
--train_data_dir ./my_dataset/
```

两者在脚本内部的加载逻辑（`train_text_to_image.py:737`）：

```python
if args.dataset_name is not None:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        data_dir=args.train_data_dir,
    )
else:
    data_files = {"train": os.path.join(args.train_data_dir, "**")}
    dataset = load_dataset(
        "imagefolder",       # HuggingFace datasets 的内置 imagefolder 加载器
        data_files=data_files,
        cache_dir=args.cache_dir,
    )
```

### 2.2 本地数据集目录结构（ImageFolder 格式）

```
my_dataset/
├── image_001.jpg
├── image_002.png
├── image_003.jpg
└── metadata.jsonl          ← 必须存在，否则 load_dataset("imagefolder") 无法读取说明
```

**`metadata.jsonl` 格式**（每行一个 JSON 对象）：

```jsonl
{"file_name": "image_001.jpg", "text": "a dog sitting on a bench in the park"}
{"file_name": "image_002.png", "text": "a cat sleeping on a sofa"}
{"file_name": "image_003.jpg", "text": ["a person walking", "a man strolling"]}
```

字段说明：

| 字段 | 类型 | 是否必需 | 说明 |
|------|------|---------|------|
| `file_name` | string | 必需 | 图像文件名，与目录中文件名精确匹配 |
| `text`（或自定义列名）| string 或 list[string] | 必需 | 图像说明；为列表时训练中随机选一条 |

> 列名可通过 `--image_column` 和 `--caption_column` 自定义，默认分别为 `"image"` 和 `"text"`。

### 2.3 数据预处理 Transform Pipeline

定义位置：`train_text_to_image.py:800`

```python
train_transforms = transforms.Compose([
    transforms.Resize(args.resolution, interpolation=interpolation),
    # center_crop=True 时确定性裁剪，否则随机裁剪
    transforms.CenterCrop(args.resolution) if args.center_crop
        else transforms.RandomCrop(args.resolution),
    transforms.RandomHorizontalFlip() if args.random_flip
        else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # 输出值域 [-1, 1]，与 vae.encode 期望一致
])
```

**输出 `pixel_values`**：`float32 Tensor`，shape `(3, resolution, resolution)`，值域 `[-1, 1]`

### 2.4 Caption 标记化

定义位置：`train_text_to_image.py:782`

```python
def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # 训练时随机选一条，验证时选第一条
            captions.append(random.choice(caption) if is_train else caption[0])
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,   # CLIP 默认 77
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids
```

### 2.5 Collate 函数与 DataLoader

定义位置：`train_text_to_image.py:829`

```python
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.train_batch_size,          # 默认 16
    num_workers=args.dataloader_num_workers,   # 默认 0
)
```

**训练循环中每个 batch 的 tensor shape**：

```python
batch = {
    "pixel_values": Tensor,    # (B, 3, resolution, resolution)，float32，[-1, 1]
    "input_ids":    Tensor,    # (B, 77)，int64
}
```

### 2.6 关键命令行参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | `None` | Hub dataset 名，与 `train_data_dir` 二选一 |
| `--train_data_dir` | `None` | 本地数据文件夹（需含 `metadata.jsonl`） |
| `--image_column` | `"image"` | dataset 中图像列名 |
| `--caption_column` | `"text"` | dataset 中说明列名 |
| `--resolution` | `512` | 图像 resize 目标分辨率 |
| `--center_crop` | `False` | True=中心裁剪，False=随机裁剪 |
| `--random_flip` | `False` | 随机水平翻转 |
| `--image_interpolation_mode` | `"lanczos"` | resize 插值方法 |
| `--train_batch_size` | `16` | 每设备 batch 大小 |
| `--dataloader_num_workers` | `0` | DataLoader 并行工作进程数 |
| `--max_train_samples` | `None` | 限制训练数据量（调试用） |

---

## 3. dreambooth 系列：数据格式与 DataLoader

### 3.1 数据目录结构

dreambooth 不需要 metadata 文件，直接读取文件夹内所有图像：

```
instance_data_dir/          ← --instance_data_dir 指定
├── photo_001.jpg
├── photo_002.jpg
└── photo_003.png

class_data_dir/             ← --class_data_dir 指定（可选，用于先验保留）
├── class_000.jpg
├── class_001.jpg
└── ...
```

**文件格式要求**：任何 PIL 可读格式（jpg、png、webp 等），自动 `convert("RGB")`。EXIF 方向信息通过 `exif_transpose()` 自动修正。

### 3.2 `DreamBoothDataset` 类

定义位置：`dreambooth/train_dreambooth.py:611`

```python
class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,      # str，实例图像文件夹路径
        instance_prompt,         # str，如 "a photo of sks dog"
        tokenizer,
        class_data_root=None,    # str | None，类别图像文件夹
        class_prompt=None,       # str | None
        class_num=None,          # int | None，最多使用多少张类别图像
        size=512,
        center_crop=False,
        encoder_hidden_states=None,        # 预计算的实例 prompt 嵌入（可选）
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        # 读取所有图像路径
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)

        # 数据集长度 = max(实例数, 类别数)，保证两边都循环到
        if class_data_root is not None:
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self._length = self.num_instance_images

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # 值域 [-1, 1]
        ])
```

**`__getitem__` 返回结构**（无先验保留）：

```python
{
    "instance_images":      Tensor,  # (3, size, size)，float32，[-1, 1]
    "instance_prompt_ids":  Tensor,  # (1, max_length)，int64
    "instance_attention_mask": Tensor,  # (1, max_length)，int64（可选）
}
```

**有先验保留时额外包含**：

```python
{
    ...
    "class_images":         Tensor,  # (3, size, size)
    "class_prompt_ids":     Tensor,  # (1, max_length)
    "class_attention_mask": Tensor,  # (1, max_length)（可选）
}
```

### 3.3 Collate 函数与先验保留的 shape 变化

定义位置：`dreambooth/train_dreambooth.py:710`

```python
def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    input_ids    = [example["instance_prompt_ids"] for example in examples]

    if with_prior_preservation:
        # 将类别图像追加到 batch 末尾，batch size 翻倍
        pixel_values += [example["class_images"] for example in examples]
        input_ids    += [example["class_prompt_ids"] for example in examples]

    pixel_values = torch.stack(pixel_values)            # (B 或 2B, 3, size, size)
    input_ids    = torch.cat(input_ids, dim=0)          # (B 或 2B, max_length)

    return {"pixel_values": pixel_values, "input_ids": input_ids}
```

> **注意**：开启 `--with_prior_preservation` 时，训练循环接收到的 `pixel_values` shape 为 `(2*B, 3, H, W)`，loss 在内部按前半/后半分别计算 diffusion loss 和 prior loss。

DataLoader 配置（`train_dreambooth.py:1109`）：

```python
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,          # 默认 4
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    num_workers=args.dataloader_num_workers,   # 默认 0
)
```

### 3.4 关键命令行参数汇总

| 参数 | 默认值 | 是否必需 | 说明 |
|------|--------|---------|------|
| `--instance_data_dir` | — | **必需** | 实例（目标对象）图像文件夹 |
| `--instance_prompt` | — | **必需** | 含标识符的提示，如 `"a photo of sks dog"` |
| `--class_data_dir` | `None` | 可选 | 类别图像文件夹（先验保留用） |
| `--class_prompt` | `None` | 与 class_data_dir 配套 | 类别提示，如 `"a photo of dog"` |
| `--with_prior_preservation` | `False` | 可选 | 开启先验保留损失 |
| `--prior_loss_weight` | `1.0` | 可选 | 先验损失权重 |
| `--num_class_images` | `100` | 可选 | 先验保留所用类别图像数量 |
| `--resolution` | `512` | 可选 | 图像分辨率 |
| `--train_batch_size` | `4` | 可选 | 每设备 batch 大小 |
| `--tokenizer_max_length` | `None` | 可选 | 覆盖 tokenizer 默认最大长度 |

---

## 4. 两种路径的数据 Batch 对比

| 字段 | text_to_image | dreambooth（无先验保留） | dreambooth（有先验保留） |
|------|--------------|----------------------|----------------------|
| `pixel_values` shape | `(B, 3, H, W)` | `(B, 3, H, W)` | `(2B, 3, H, W)` |
| `input_ids` shape | `(B, 77)` | `(B, 77)` | `(2B, 77)` |
| `attention_mask` | 无 | `(B, 77)`（可选） | `(2B, 77)`（可选） |
| 值域 | `[-1, 1]` | `[-1, 1]` | `[-1, 1]` |
| dtype | `float32` | `float32` | `float32` |

---

## 5. 训练脚本数据流全链路

```
磁盘文件
    │
    ├─ metadata.jsonl + 图像文件
    │       ↓
    │  datasets.load_dataset("imagefolder")
    │       ↓
    │  dataset["train"]（HuggingFace Dataset 对象）
    │       ↓ .with_transform(preprocess_train)
    │  train_dataset（自动 transform）
    │
    └─ 图像文件夹
            ↓
       DreamBoothDataset.__init__()（Path.iterdir()）
            ↓ __getitem__()
       example dict（逐条处理）

共同路径：
    ↓
DataLoader（shuffle, collate_fn, num_workers）
    ↓
batch = {"pixel_values": Tensor, "input_ids": Tensor}
    ↓
vae.encode(pixel_values) * scaling_factor   → model_input (latent)
unet(noisy_latents, timesteps, text_embeds) → noise_pred
F.mse_loss(noise_pred, target)              → loss
```

---

## 6. 快速配置模板

### text_to_image（本地数据集）

```bash
# 1. 准备 metadata.jsonl
echo '{"file_name": "img_001.jpg", "text": "your caption here"}' >> ./data/metadata.jsonl

# 2. 启动训练
accelerate launch examples/text_to_image/train_text_to_image.py \
  --train_data_dir ./data \
  --image_column image \
  --caption_column text \
  --resolution 512 \
  --random_flip \
  --train_batch_size 4 \
  --dataloader_num_workers 4
```

### dreambooth（小样本微调）

```bash
# 数据：5-10 张目标对象图像放在一个文件夹中即可，无需 metadata
accelerate launch examples/dreambooth/train_dreambooth.py \
  --instance_data_dir ./my_images \
  --instance_prompt "a photo of sks dog" \
  --class_data_dir ./class_images \         # 可选
  --class_prompt "a photo of dog" \         # 可选
  --with_prior_preservation \               # 可选
  --resolution 512 \
  --train_batch_size 1
```
