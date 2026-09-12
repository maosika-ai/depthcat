# Contributing

Thanks for looking. Small, focused PRs are easiest to review.

## Setup

```bash
git clone git@github.com:maosika-ai/reshot.git && cd reshot
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision            # pick the build for your machine
pip install -e ".[ffmpeg,dev]"
```

## Before you push

```bash
ruff check reshot tests && ruff format reshot tests
pytest -q                                # ~30 tests, no model, ~2 s
RUN_MODEL_TESTS=1 pytest -q tests/test_model_smoke.py   # optional: real model on CPU, downloads 111 MB
```

CI runs the same on Ubuntu and Windows, plus the model smoke on CPU.

## Ground rules

- `reshot/third_party/` is vendored upstream code — do not edit it; fix things in our layer.
- User-facing failures raise a `ReshotError` subclass with a concrete fix in the message.
- Anything that changes output (normalisation, encoding, presets) needs a test and a
  CHANGELOG line, and the README/USAGE tables updated in **both** languages.
- Performance claims in docs must come from a measurement; say where it was measured.

## Reporting a bug

Include: OS, GPU/CPU, `python -c "import torch;print(torch.__version__)"`, the exact command,
and the stderr output with `-v`. If a clip reproduces it and you can share it, link it.
