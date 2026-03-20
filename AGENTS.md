# Repository Guidelines

## Project structure & module organization

- `src/diffusers/`: library code (`models/`, `pipelines/`, `schedulers/`, `utils/`, `modular_pipelines/`).
- `tests/`: pytest suite (organized by area: `pipelines/`, `models/`, `lora/`, etc.).
- `examples/`: runnable training/inference scripts (some are tested via `make test-examples`).
- `docs/`: documentation sources under `docs/source/` (see `docs/README.md` for local preview).
- `utils/`, `scripts/`, `benchmarks/`: maintenance scripts, repo checks, and benchmarking helpers.

## Build, test, and development commands

Set up a virtualenv and install in editable mode:

```bash
pip install -e ".[dev]"
```

Common Makefile targets (note: Makefile sets `PYTHONPATH=src` to use your local checkout):

```bash
make quality        # lint/format checks + doc style + toc checks
make style          # auto-fix + regenerate dependency table + extra checks
make fixup          # fast fix on modified files + repo consistency
make test           # pytest ./tests (xdist)
make test-examples  # pytest ./examples
```

Docs preview:

```bash
pip install -e ".[docs]" && pip install git+https://github.com/huggingface/doc-builder
doc-builder preview diffusers docs/source/en
```

## Coding style & naming conventions

- Python is formatted/linted with Ruff (`line-length = 119`, spaces, double quotes). Prefer `make style` over manual fixes.
- Keep changes explicit and readable; pipelines/schedulers favor self-contained (“single-file”) implementations over deep abstractions (see `PHILOSOPHY.md`).

## Testing guidelines

- Add/adjust tests next to the feature area (for example, a pipeline change usually needs a test under `tests/pipelines/`).
- Run focused tests first: `python -m pytest tests/pipelines/test_x.py -k keyword`.
- If you add slow tests, ensure they pass with `RUN_SLOW=1 python -m pytest ...` when applicable.

## Commit & pull request guidelines

- Commit/PR titles are typically imperative and may include a scope tag (for example, `docs: ...`, `[Modular] ...`) and a PR number `(#NNNNN)`.
- Keep PRs laser-focused, link related issues, include a short usage snippet when helpful, and add high-coverage tests.
- Avoid adding large binaries (images/videos). Prefer Hugging Face-hosted datasets for assets referenced by tests/docs.
