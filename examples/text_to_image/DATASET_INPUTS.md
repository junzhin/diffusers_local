# Text-to-Image 数据输入格式（`examples/text_to_image`）

本文档精确描述本仓库 Text-to-Image 训练脚本对“数据集输入”的要求，目标是让你能据此直接改代码（列名/格式/加载方式/分割等）。

## 适用脚本

- SD 1.x/2.x：`examples/text_to_image/train_text_to_image.py`、`examples/text_to_image/train_text_to_image_lora.py`
- SDXL：`examples/text_to_image/train_text_to_image_sdxl.py`、`examples/text_to_image/train_text_to_image_lora_sdxl.py`

四个脚本共同约定：数据集中必须有 **一列图片** 和 **一列文本 caption**。默认列名分别是 `image` 和 `text`，可通过 CLI 覆盖。

## 数据入口（二选一）

### A. 通过 `datasets.load_dataset()` 加载（Hub 数据集或本地 datasets 目录）

- 必填：`--dataset_name <name-or-path>`
- 可选：`--dataset_config_name <config>`
- 注意：脚本会把 `--train_data_dir` 传给 `load_dataset(..., data_dir=...)`（例如 `train_text_to_image.py:739`、`train_text_to_image_sdxl.py:834`）。
- **硬性要求：必须存在 `train` split**，因为脚本直接访问 `dataset["train"]`（例如 `train_text_to_image.py:759`）。

### B. 通过本地文件夹加载（datasets 的 `imagefolder` builder）

- 必填：不传 `--dataset_name`，传 `--train_data_dir /abs/or/rel/path`
- 脚本行为：
  - `load_dataset("imagefolder", data_files={"train": os.path.join(train_data_dir, "**")})`（例如 `train_text_to_image.py:749`）
- **硬性要求：`imagefolder` 必须能把该目录解析成一个包含 `train` split 的 dataset**。

## 必需列（columns）与类型（types）

### 列名如何指定

- `--image_column`：默认 `image`
- `--caption_column`：默认 `text`

脚本会检查列是否存在（例如 `train_text_to_image.py:766`、`train_text_to_image.py:774`）。

### 图片列（`image_column`）的类型要求

每条样本的图片对象必须满足：

- 支持 `.convert("RGB")`（datasets 通常提供 `PIL.Image.Image`）
- 图片可以是任意原始分辨率/模式；训练前会统一做 Resize/Crop/Flip/Normalize

数据预处理（所有脚本都类似）：

- Resize 到 `--resolution`
- Crop：`--center_crop` 为真则中心裁剪；否则随机裁剪
- 可选：`--random_flip` 随机水平翻转
- `ToTensor()` 并 `Normalize([0.5],[0.5])`

参考实现：`train_text_to_image.py:807`，`train_text_to_image_sdxl.py:872`。

### 文本列（`caption_column`）的类型要求

每条样本的 caption 必须是以下之一：

- `str`：单条 caption
- `list[str]` 或 `np.ndarray[str]`：多条 caption（训练时随机取一条；验证/非训练时取第 1 条）

参考实现：

- SD 1.x/LoRA：`tokenize_captions()`（`train_text_to_image.py:782`、`train_text_to_image_lora.py:650`）
- SDXL LoRA：`tokenize_captions()`（`train_text_to_image_lora_sdxl.py:909`）
- SDXL（全参脚本）：`encode_prompt()`（`train_text_to_image_sdxl.py:502`）

## 本地数据集推荐格式：imagefolder + `metadata.jsonl`

一个最小可用结构（图片可放在子目录）：

```text
my_data/
  metadata.jsonl
  0001.jpg
  0002.png
  subdir/0003.webp
```

`metadata.jsonl`：每行一个 JSON，对应一张图片。示例（字段名与默认列名对齐）：

```jsonl
{"file_name":"0001.jpg","text":"a photo of a corgi"}
{"file_name":"subdir/0003.webp","text":["caption a","caption b"]}
```

约束/建议：

- `file_name` 通常是相对 `my_data/` 的路径（保持稳定、可复现）。
- caption 字段名必须与 `--caption_column` 一致；若你用 `caption`/`prompt` 等字段名，启动脚本时要传 `--caption_column=caption`。
- 若图片列名不是 `image`（极少见，通常不用改），传 `--image_column=...`。

## SDXL 脚本的“列名陷阱”（会影响你改数据格式）

`train_text_to_image_sdxl.py` 会先预计算并缓存 prompt embedding 和 VAE latent（`dataset.map(...)`），然后把两个数据集合并。

- 目前合并时写死了 `remove_columns(["image", "text"])`（见 `train_text_to_image_sdxl.py:945`）

因此：

- 如果你通过 CLI 改了列名（例如 `--caption_column=caption`），脚本会在这里报错（找不到 `text`/`image`）。

要支持自定义列名，需要改代码：

- 把 `remove_columns(["image", "text"])` 改为使用前面已经解析并校验过的 `image_column` / `caption_column` 变量（它们在 `train_text_to_image_sdxl.py:855` 附近确定）。

## 典型改动需求与代码定位

- 数据集没有 `train` split：修改 `dataset["train"]` 的访问位置（如 `train_text_to_image.py:759`、`train_text_to_image_sdxl.py:851`）。
- caption 不是 `str` 或 `list[str]`：修改 `tokenize_captions()` / `encode_prompt()` 的解析逻辑（如 `train_text_to_image.py:782`、`train_text_to_image_sdxl.py:502`）。
- 想用自定义数据加载逻辑（不走 `imagefolder`）：修改 `load_dataset(...)` 的那段（如 `train_text_to_image.py:737`、`train_text_to_image_sdxl.py:832`）。
- SDXL 想自定义列名：除 `--image_column/--caption_column` 外，必须修 `train_text_to_image_sdxl.py:945` 的 `remove_columns(...)`。
