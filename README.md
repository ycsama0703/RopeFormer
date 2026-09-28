# RopeFormer

An encoder-only Transformer for financial time series in which **each attention
head is given its own positional time scale**, implemented by scaling the
frequency of the rotary position embedding (RoPE) per head. The model emits a
direction logit, a conditional mean, and a conditional log-volatility for every
step of the forecast horizon.

This repository contains **the method only**. It is the shared starting point
for our group project; the dataset, training loop, and experiments are
deliberately out of scope for now and will be added later.

---

## Scope

| In scope | Out of scope (for now) |
| --- | --- |
| The model definition (`src/ropeformer/`) | Datasets and loaders |
| A method registry | Training loop / optimizer / losses |
| Diagnostic and smoke-test scripts | Baselines, metrics, evaluation |
| Method documentation (`docs/`) | Experiment configs and results |

The code was lifted verbatim from an earlier single-asset experiment so that we
all start from the same baseline. It is **not** clean and it has confirmed
defects — see [`docs/03-known-issues.md`](docs/03-known-issues.md) before you
build on it. We expect to partially redesign the method, so treat the current
implementation as a reference point, not as a foundation to preserve.

---

## Layout

```
RopeFormer/
├── README.md
├── requirements.txt
├── docs/
│   ├── 01-method.md              Full specification of the current method
│   ├── 02-related-work.md        Where this sits relative to prior work
│   ├── 03-known-issues.md        Confirmed defects, with reproductions
│   └── 04-redesign-directions.md Candidate directions for the redesign
├── src/ropeformer/
│   ├── __init__.py
│   ├── ropeformer.py             The model (verbatim from the original repo)
│   └── registry.py               Name -> class registry
└── tools/
    ├── smoke_test.py             Forward pass on random tensors
    └── check_rope_spectrum.py    RoPE frequency / scaling diagnostics
```

## Setup

```bash
pip install -r requirements.txt
python tools/smoke_test.py
```

Expected output ends with `OK` and prints the three output tensors at shape
`[batch, horizon]`.

```bash
python tools/check_rope_spectrum.py
```

This one is **expected to report FAIL** on the current code. That is not a
broken install — it is reproducing a real defect described in
[`docs/03-known-issues.md`](docs/03-known-issues.md).

## Usage

```python
import sys; sys.path.insert(0, "src")
import torch
from ropeformer import RopeFormer

model = RopeFormer(
    d_model=64, nhead=4, enc_layers=2, d_ff=128,
    horizon=5, in_dim=8,
    head_freq_scales=[0.25, 0.5, 1.0, 2.0],
    learnable_head_scaling=True,
    use_cross_attn=True,
)

out = model(
    x_enc=torch.randn(4, 60, 8),   # encoder window
    x_dec=torch.randn(4, 8, 8),    # interface only, not used by the math
    x_mem=torch.randn(4, 60, 12),  # exogenous factors, for cross-attention
)
# out["mu"], out["log_sigma"], out["logits"] -> each [4, 5]
```

Pass `d_ff` explicitly. Do **not** pass `ff_mult` as an integer — see issue 3 in
the known-issues document.

## Where to start reading

1. [`docs/01-method.md`](docs/01-method.md) — what the model actually computes.
2. [`docs/03-known-issues.md`](docs/03-known-issues.md) — what is broken in it.
3. [`docs/04-redesign-directions.md`](docs/04-redesign-directions.md) — the
   options we are choosing between.
4. [`docs/02-related-work.md`](docs/02-related-work.md) — needed for the
   proposal's positioning, and for honest citation.

## Provenance

Source: `oil_index/code/methods/ropeformer.py` (repo `ycsama0703/oil_index`).
Inline code comments are the original author's and are partly in Chinese; all
documentation in `docs/` is in English.
