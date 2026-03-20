# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Install
```bash
pip install -e ".[dev]"   # Full dev environment (includes quality + test deps)
pip install -e ".[torch]" # Inference-only with PyTorch
```

### Code Quality
```bash
make quality              # Check only (Ruff lint + format, no writes)
make style                # Auto-fix + format all files
make fixup                # Quick fix: only files changed vs main branch
make modified_only_fixup  # Fix only files changed in current branch
```

### Testing
```bash
# Full test suite (parallel across CPU cores, balanced by file)
python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Single test file
python -m pytest tests/models/test_modeling_vae.py -v

# Single test method
python -m pytest tests/models/test_modeling_vae.py::AutoencoderKLTests::test_forward_signature -v

# Environment variable gates (off by default):
RUN_SLOW=1    python -m pytest tests/ -v   # tests decorated with @slow
RUN_NIGHTLY=1 python -m pytest tests/ -v   # tests decorated with @nightly
RUN_COMPILE=1 python -m pytest tests/ -v   # torch.compile tests
```

Test decorators are defined in `src/diffusers/utils/testing_utils.py`. The most common ones:
- `@slow` — requires `RUN_SLOW=1`; skipped in CI by default
- `@require_torch_gpu` — skipped if no CUDA
- `@require_torch_multi_gpu` — skipped unless `torch.cuda.device_count() > 1`
- `@skip_mps` — skip on Apple Silicon MPS backend
- `@require_flax`, `@require_onnxruntime` — skip if optional backend absent

### Repo Consistency (run before submitting a PR)
```bash
make repo-consistency        # Runs all checks below
python utils/check_dummies.py    # Validate stub classes for optional deps
python utils/check_inits.py      # Validate __init__.py export lists
python utils/check_copies.py     # Validate "# Copied from" marker consistency
```

### After Modifying `setup.py` Dependencies
```bash
make deps_table_update  # Regenerates src/diffusers/dependency_versions_table.py
```

---

## Architecture Overview

### Three Independent, Swappable Layers

```
Pipeline  (end-to-end inference workflow)
    └── Model components: vae, unet/transformer, text_encoder, ...
    └── Scheduler        (noise schedule + denoising step logic)
```

Each layer can be loaded, saved, and replaced independently via `from_pretrained()` / `save_pretrained()`.

---

### Pipelines (`src/diffusers/pipelines/`)

Every pipeline lives in its own folder (e.g., `stable_diffusion/`), one Python file per variant. All inherit from `DiffusionPipeline`.

**How pipeline components are discovered and loaded:**

`DiffusionPipeline.from_pretrained()` reads `model_index.json` from the checkpoint directory. That file maps attribute names to `[library, ClassName]` pairs:

```json
{
  "_class_name": "StableDiffusionPipeline",
  "vae": ["diffusers", "AutoencoderKL"],
  "unet": ["diffusers", "UNet2DConditionModel"],
  "scheduler": ["diffusers", "PNDMScheduler"],
  "tokenizer": ["transformers", "CLIPTokenizer"],
  "text_encoder": ["transformers", "CLIPTextModel"]
}
```

Each class listed must be a subclass of a type registered in `LOADABLE_CLASSES` (defined in `src/diffusers/pipelines/pipeline_loading_utils.py`). That registry controls which base classes can be pipeline components:

- `diffusers`: `ModelMixin`, `SchedulerMixin`, `DiffusionPipeline`, `BaseGuidance`
- `transformers`: `PreTrainedModel`, `PreTrainedTokenizer`, `ProcessorMixin`, etc.

This means **any custom class that inherits `ModelMixin` is automatically a loadable pipeline component** — no registration needed beyond the `model_index.json` entry.

**Replacing a pipeline component at runtime:**
```python
pipe = StableDiffusionPipeline.from_pretrained("sd-v1-5")
pipe.vae = MyCustomVAE.from_pretrained("my-vae-checkpoint")
# or assign any nn.Module implementing .encode() / .decode()
```

**Skipping the decoder at inference time** (all major pipelines support this):
```python
latents = pipe("a photo", output_type="latent").images  # returns raw latent tensor
```
The pipeline checks `if not output_type == "latent": image = self.vae.decode(...)` and skips decoding entirely.

---

### Models (`src/diffusers/models/`)

Subdirectories: `unets/`, `transformers/`, `autoencoders/`, `controlnets/`. Shared building blocks live at the top level: `attention.py`, `resnet.py`, `embeddings.py`, `normalization.py`.

**All models inherit `ModelMixin` + `ConfigMixin`.** This provides:
- `from_pretrained()` / `save_pretrained()` with automatic `config.json` serialization
- `enable_gradient_checkpointing()`, device/dtype casting, quantization hooks

**`@register_to_config` decorator** (defined in `configuration_utils.py`):

Applied to every model's `__init__`. It intercepts all constructor arguments and stores them in `self.config` automatically. This means:
- Every `__init__` kwarg is serialized to `config.json` on save
- `from_pretrained()` reconstructs the model by re-calling `__init__` with the saved config
- Adding a new `__init__` kwarg with a default value is backward-compatible with old checkpoints

**`@apply_forward_hook` decorator** (`src/diffusers/utils/accelerate_utils.py`):

Used on `.encode()` and `.decode()` methods of autoencoders. When CPU offloading is active (accelerate hooks), PyTorch only triggers device movement on `.forward()`. This decorator makes `.encode()` and `.decode()` also trigger the offload hooks correctly.

**Autoencoder variants** (`src/diffusers/models/autoencoders/`):

| Class | Notes |
|-------|-------|
| `AutoencoderKL` | Standard KL-VAE. Default for SD/SDXL/Flux. `encoder`, `decoder` are separate `nn.Module` attributes. `quant_conv`/`post_quant_conv` can be disabled via `use_quant_conv=False`. |
| `VQModel` | VQ-VAE. Contains `encoder`, `quantize` (VectorQuantizer), `decoder`. `force_not_quantize=True` in `.decode()` skips the codebook lookup. |
| `AutoencoderTiny` | TAESD. Lightweight fast decoder, no quantization. |
| `AutoencoderDC` | DC-AE (SANA). |
| `AutoencoderKLMagvit` | MAGViT-style VQ tokenizer. |

Internal building blocks shared by all autoencoders live in `src/diffusers/models/autoencoders/vae.py`: `Encoder`, `Decoder`, `VectorQuantizer`, `DiagonalGaussianDistribution`, `EncoderTiny`, `DecoderTiny`.

---

### Schedulers (`src/diffusers/schedulers/`)

One file per algorithm. All inherit `SchedulerMixin` + `ConfigMixin`.

Required interface every scheduler must implement:
- `set_timesteps(num_inference_steps)` — must be called before each denoising loop
- `step(model_output, timestep, sample)` — returns `prev_sample`
- `timesteps` attribute — the array the denoising loop iterates over

Schedulers are interchangeable via `ConfigMixin.from_config()`:
```python
new_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = new_scheduler
```

---

### Key Design Patterns

**Single-file policy (pipelines + schedulers):** Each pipeline/scheduler is intentionally self-contained. Code duplication is deliberate and enforced. `# Copied from diffusers.pipelines.X.Y` comments mark intentional copies; `utils/check_copies.py` validates they stay in sync with the source. Never refactor shared logic into a utility unless it belongs in `src/diffusers/utils/`.

**Lazy imports + dummy stubs:** `src/diffusers/__init__.py` uses `_LazyModule` to avoid importing all backends at startup. For every class that depends on an optional package (torch, flax, transformers, etc.), a corresponding stub must exist in `src/diffusers/utils/dummy_*.py`. Run `python utils/check_dummies.py` to validate. When adding a new public class:
1. Add it to `src/diffusers/__init__.py` under the correct `_import_structure` entry
2. Add a stub in the appropriate `src/diffusers/utils/dummy_*.py`
3. Run `make repo-consistency` to verify

**`examples/` training scripts:** These are standalone scripts, not part of the installed library. They import from `diffusers` but are not tested with `make test`. They demonstrate the canonical pattern for latent diffusion training:
- `vae = None` → trains directly in pixel space (UNet gets raw images)
- `vae is not None` → encodes with `vae.encode().latent_dist.sample() * scaling_factor`, trains in latent space
- `vae.requires_grad_(False)` → VAE is frozen during UNet training (standard practice)

---

## Adding New Components

### New Pipeline
1. Create `src/diffusers/pipelines/<name>/pipeline_<name>.py`
2. Inherit `DiffusionPipeline`, implement `__call__` with a `return_dict` arg
3. Add to `src/diffusers/pipelines/__init__.py` and `src/diffusers/__init__.py`
4. If it depends on optional backends, add stubs to `src/diffusers/utils/dummy_*.py`
5. Run `make repo-consistency`

### New Model
1. Create `src/diffusers/models/<subdir>/<name>.py`
2. Inherit `ModelMixin` + `ConfigMixin`, decorate `__init__` with `@register_to_config`
3. Add to `src/diffusers/models/__init__.py` and `src/diffusers/__init__.py`
4. Follow the same stub + consistency check steps as pipelines

### New Scheduler
1. Create `src/diffusers/schedulers/scheduling_<name>.py` (single-file, no imports from large utils)
2. Inherit `SchedulerMixin` + `ConfigMixin`, implement `set_timesteps` + `step` + `timesteps`
3. Add to `src/diffusers/schedulers/__init__.py` and `src/diffusers/__init__.py`

---

## Code Style

- Line length: 119 characters (Ruff)
- Quotes: double (Black-compatible)
- Imports: sorted by Ruff with `lines-after-imports = 2`; `known-first-party = ["diffusers"]`
- No lambda functions or advanced PyTorch one-liners — prefer explicit, readable code per project philosophy
