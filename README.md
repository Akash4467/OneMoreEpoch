# OneMoreEpoch

[![Latest Release](https://img.shields.io/github/v/release/Akash4467/OneMoreEpoch?label=release)](https://github.com/Akash4467/OneMoreEpoch/releases)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![CI](https://github.com/Akash4467/OneMoreEpoch/actions/workflows/ci.yml/badge.svg)](https://github.com/Akash4467/OneMoreEpoch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Akash4467/OneMoreEpoch/blob/main/LICENSE)

**A deep learning framework you can actually read.** Built in Python for
everyday use, with a Rust engine underneath for speed, a real automatic
differentiation system, and an optional Hindi/Hinglish personality that
roasts you when your code breaks.

It's not trying to replace PyTorch or TensorFlow. It's built to be
understood — every layer of it — while still being genuinely usable for
training real models.

```bash
pip install onemoreepoch
```

---

## What this actually is

Training a neural network means: run some numbers through a model, see how
wrong the answer was, and nudge every internal number a tiny bit to be less
wrong next time. Do that thousands of times and the model gets good at its
job.

OneMoreEpoch is the machinery that makes that loop work, split into pieces
that each do one job well:

- **Tensors** are the basic unit of data here — think of them as
  spreadsheets of numbers (a photo, a batch of text, a set of measurements)
  that the framework can do math on quickly.
- **Autograd** (automatic differentiation) is the part that automatically
  figures out *which direction* to nudge every number in the model so it
  gets better, without a human having to work out that math by hand for
  every new model shape.
- **Layers and optimizers** are the reusable building blocks (like
  "a fully connected layer" or "the Adam update rule") you snap together to
  build and train a model, instead of writing the math from scratch every
  time.
- **A Python front door, a Rust engine room.** You write plain Python. Under
  the hood, the actual number-crunching can run through a compiled Rust
  engine instead of Python — same results, much faster, because Rust runs
  closer to the metal than Python does.
- **A personality system.** When something goes wrong (and in machine
  learning, something always goes wrong), the error message can talk to you
  in plain professional English, Hinglish ("bhai, ye shapes ka rishta nahi
  ho sakta"), or a savage-but-friendly roast — your choice.

None of this needs you to understand Rust, or even much Python, to use it —
those details are exactly what the framework hides from you.

---

## Installation

```bash
pip install onemoreepoch
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add onemoreepoch
```

OneMoreEpoch's faster Rust backend ships as a compiled extension. When a
pre-built wheel exists for your platform and Python version, that's all
`pip` needs — no Rust, no compiler, nothing else to install. If pip has to
build from source instead (an uncommon platform, for example), you'll need
a [Rust toolchain](https://rustup.rs/) on your machine; either way, the
default NumPy backend always works with no extra setup.

## Quick start

```python
from onemoreepoch import nn
from onemoreepoch.core import Tensor
from onemoreepoch.optim import Adam

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = Adam(model.parameters())

x = Tensor.randn(32, 10)
target = Tensor.randn(32, 1)

for _ in range(100):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), target)
    loss.backward()
    optimizer.step()
```

Want the personality system? One line:

```python
from onemoreepoch import config
config.set_message_mode("hindi")  # or "roast" — "classic" is the default
```

---

## API Reference

A guide to every public function/class — what to import, what arguments it takes, and what it does. This is what to check when you're calling something and can't remember the exact parameters.

### `Tensor` — `from onemoreepoch.core import Tensor`

The core data type. Everything else operates on Tensors.

```python
Tensor(data, *, dtype=None, device=None, requires_grad=False)
```
Wraps `data` (a list, tuple, NumPy array, or scalar) as a Tensor. Set `requires_grad=True` if you need gradients with respect to it.

| Factory | Creates |
|---|---|
| `Tensor.zeros(*shape, requires_grad=False)` | filled with `0` |
| `Tensor.ones(*shape, requires_grad=False)` | filled with `1` |
| `Tensor.randn(*shape, requires_grad=False)` | standard-normal random values |
| `Tensor.rand(*shape, requires_grad=False)` | uniform `[0, 1)` random values |

Properties: `.shape`, `.ndim`, `.dtype`, `.size`, `.grad` (gradient after `.backward()`, or `None`), `.requires_grad`, `.device`, `.T` (transpose).

| Method | Does |
|---|---|
| `.numpy()` | returns the data as a real NumPy array, regardless of backend |
| `.item()` | returns a single-element tensor's value as a plain Python number |
| `.detach()` | returns a copy that's cut off from the computation graph |
| `.backward(grad=None)` | computes gradients; `grad` is only needed for non-scalar tensors |
| `.zero_grad()` | clears `.grad` back to `None` |
| `.reshape(*shape)` | returns a reshaped view |
| `.transpose(*axes)` | returns axes permuted (no args = full reversal) |
| `.sum(axis=None, keepdims=False)` | sums over an axis (or everything) |
| `.mean(axis=None, keepdims=False)` | averages over an axis (or everything) |
| `.exp()` / `.log()` | elementwise `e^x` / `ln(x)` |

Operators `+ - * / @ **` and unary `-` all work between Tensors (and plain numbers), and are all differentiable.

### `nn` — `from onemoreepoch import nn`

**`nn.Module`** — base class for anything you build. Subclass it and implement `forward(self, x)`.

| Method | Does |
|---|---|
| `.parameters()` | list of all trainable `Parameter`s, recursively |
| `.named_parameters()` | same, as `(dotted.name, parameter)` pairs |
| `.zero_grad()` | clears every parameter's gradient |
| `.train()` / `.eval()` | switches training/inference mode (affects `Dropout`, `BatchNorm`) |
| `.state_dict()` / `.load_state_dict(state)` | save/restore parameter values by name |

**Layers:**

| Layer | Signature | Does |
|---|---|---|
| `nn.Linear` | `(in_features, out_features, *, bias=True)` | `y = x @ W + b` |
| `nn.Conv2D` | `(in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True)` | 2-D convolution |
| `nn.Dropout` | `(p=0.5)` | zeroes elements with probability `p` during training only |
| `nn.BatchNorm` | `(num_features, *, eps=1e-5, momentum=0.1)` | normalizes a `(N, C)` batch to zero mean / unit variance |
| `nn.Sequential` | `(*modules)` | runs modules in order; index/iterate it like a list |

**Activations** (no arguments): `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()`.

**Losses**: `nn.MSELoss()` — call as `criterion(prediction, target)`, returns mean squared error.

**`nn.init`** — weight initializers, each mutates a Tensor in place and returns it:

| Function | Does |
|---|---|
| `init.zeros_(tensor)` / `init.ones_(tensor)` | fills with `0` / `1` |
| `init.uniform_(tensor, low=0.0, high=1.0)` | fills from `Uniform[low, high)` |
| `init.normal_(tensor, mean=0.0, std=1.0)` | fills from `Normal(mean, std)` |
| `init.kaiming_uniform_(tensor, fan_in=None)` | `Uniform[-b, b]`, `b = fan_in ** -0.5` |
| `init.xavier_uniform_(tensor, fan_in=None, fan_out=None)` | `Uniform[-b, b]`, `b = sqrt(6 / (fan_in + fan_out))` |

### `optim` — `from onemoreepoch import optim`

Every optimizer takes `model.parameters()` first, then hyperparameters, and exposes `.step()` (apply one update) and `.zero_grad()` (clear gradients before the next backward pass).

| Optimizer | Signature |
|---|---|
| `optim.SGD` | `(parameters, lr=0.01, momentum=0.0)` |
| `optim.Adam` | `(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8)` |
| `optim.AdamW` | `(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` |
| `optim.RMSProp` | `(parameters, lr=0.01, alpha=0.99, eps=1e-8, momentum=0.0)` |
| `optim.AdaGrad` | `(parameters, lr=0.01, eps=1e-10)` |

### `data` — `from onemoreepoch import data`

| | Signature | Does |
|---|---|---|
| `data.Dataset` | — | abstract base: implement `__len__` and `__getitem__` for your own dataset |
| `data.TensorDataset` | `(*arrays)` | wraps parallel arrays (e.g. features, labels) as a `Dataset` of tuples |
| `data.DataLoader` | `(dataset, batch_size=1, *, shuffle=False, seed=None)` | iterate it for batches; each batch mirrors the shape `__getitem__` returns |

### `metrics` — `from onemoreepoch import metrics`

Each takes `(predictions, targets)` and returns a plain float. Accepts Tensors, NumPy arrays, or plain lists.

| Function | Does |
|---|---|
| `metrics.mse` / `metrics.mae` | mean squared / absolute error |
| `metrics.accuracy` | fraction of matching predicted classes (auto-detects logits vs. class indices) |
| `metrics.precision(*, positive_label=1)` | of predicted positives, fraction actually positive |
| `metrics.recall(*, positive_label=1)` | of actual positives, fraction correctly predicted |
| `metrics.f1(*, positive_label=1)` | harmonic mean of precision and recall |

### `config` — `from onemoreepoch import config`

| Function | Does |
|---|---|
| `config.set_message_mode(mode)` | `"classic"` (default), `"hindi"`, or `"roast"` |
| `config.get_message_mode()` | returns the current mode |
| `config.set_debug_checks(enabled)` | turns gradient explosion/vanishing warnings on or off |
| `config.debug_checks_enabled()` | returns whether they're on |

Also settable via environment variables before import: `ONEMOREEPOCH_MESSAGES=hindi`, `EDUCATIONAL_MODE=1` (shorthand for hindi).

### Backends — `from onemoreepoch.core.backend import get_backend`

| Function | Does |
|---|---|
| `get_backend(name=None)` | returns the backend instance (`"numpy"` default, or `"rust"`) |
| `register_backend(backend)` | registers a custom `Backend` implementation |

Also settable via `ONEMOREEPOCH_BACKEND=rust` before import, to make Rust the default without code changes.

---

## How the pieces fit together

```
Your code (Python)
        │
        ▼
 Tensor / nn / optim      ← the public API you actually write
        │
        ▼
   Autograd engine        ← figures out how to improve the model
        │
        ▼
  Backend (pluggable)     ← does the actual math
        │
   ┌────┴────┐
   ▼         ▼
 NumPy      Rust           ← two interchangeable "engines"; pick
(default)  (faster)          whichever one is available/faster
```

The important idea is the **backend**: the framework never does math
directly, it always asks "whichever backend is active" to do it. That's
what let a whole Rust engine get added *without changing a single line* of
the training code, layers, or optimizers above it — and it's the same seam
GPU support will plug into next (see Roadmap below).

### The personality system, and why it's not a gimmick

Error messages in most tools are dry and unhelpful. Here, every error is
backed by a real, structured, machine-readable exception underneath — the
personality layer only changes the *words shown to you*, never what
actually happened or how the program behaves. So `bhai, ye shapes ka rishta
nahi ho sakta` and `ShapeError: matrix dimensions must agree` are the exact
same error, just spoken differently. A joke can never hide a real problem.

---

## Project layout

```
OneMoreEpoch/
├── src/onemoreepoch/
│   ├── core/            Tensor, Module, Parameter, Device, and backend/
│   │   └── backend/         the pluggable math engines (NumPy, Rust, ...)
│   ├── autograd/         the "figure out how to improve" engine
│   ├── nn/               layers (Linear, Conv2D, Dropout, BatchNorm, ...)
│   ├── optim/             training rules (SGD, Adam, AdamW, RMSProp, AdaGrad)
│   ├── data/              loading and batching datasets
│   ├── metrics/           accuracy, precision, recall, and friends
│   └── messages/          classic / hindi / roast personalities + memes
├── rust/onemoreepoch-core/   the compiled speed engine (PyO3 + Rust)
├── tools/meme_updater/       local pipeline for adding new personality lines
├── tests/                    unit, integration, and personality-system tests
└── examples/                  runnable end-to-end scripts
```

---

## Roadmap: what's next — GPU acceleration

The next piece of work is a **GPU backend**, so training can run on a
graphics card instead of just the CPU (typically a 10–100x speed-up for
larger models).

This is the reason the backend system was built to be pluggable in the
first place: adding the Rust engine didn't require touching `Tensor`,
`nn`, or `optim` at all — it was purely a new engine registered alongside
the NumPy one. GPU support is planned to slot in the exact same way,
which two things fall out of directly:

- **Where it plugs in**: `src/onemoreepoch/core/backend/` — a new backend
  implementation alongside `numpy_backend.py` and `rust_backend.py`.
- **What it needs to solve that's genuinely new**: unlike swapping NumPy
  for Rust, a GPU also needs a *scheduler* — something deciding which
  computations run on the GPU, when data moves between CPU and GPU memory
  (that trip is slow, so minimizing it matters), and how multiple
  operations get batched together so the GPU stays busy instead of waiting
  around for the CPU to hand it work one step at a time.

This section is intentionally short — it's a marker for the next phase of
work, not a spec written in advance of actually doing it.

---

## Development

Contributing or just poking around the source (rather than installing the
published package)?

```bash
git clone https://github.com/Akash4467/OneMoreEpoch.git
cd OneMoreEpoch
uv sync --group dev      # installs dependencies and builds the Rust engine
uv run pytest              # 200+ tests, should all pass
cargo test --no-default-features --manifest-path rust/onemoreepoch-core/Cargo.toml
```

This needs a [Rust toolchain](https://rustup.rs/) regardless of platform,
since you're building the extension yourself rather than installing a
pre-built wheel.

## Links

- [Source code](https://github.com/Akash4467/OneMoreEpoch)
- [Issue tracker](https://github.com/Akash4467/OneMoreEpoch/issues)
- [PyPI package](https://pypi.org/project/onemoreepoch/)

## License

MIT — see [LICENSE](https://github.com/Akash4467/OneMoreEpoch/blob/main/LICENSE).
