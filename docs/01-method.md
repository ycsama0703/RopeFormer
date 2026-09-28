# The Method

> Specification of the code currently in `src/ropeformer/ropeformer.py`.
> This describes what the model **does**, not what it should do. For defects,
> see [`03-known-issues.md`](03-known-issues.md).

## 1. The idea in one paragraph

In a standard Transformer every attention head receives the same positional
encoding, so every head has the same notion of how far apart two time steps
are. RopeFormer gives head `h` its own scalar `s_h` that multiplies position
before the rotary angle is computed. A head with a small `s_h` sees positions
stretched out (slow, long-range structure); a head with a large `s_h` sees them
compressed (fast, short-range structure). The scales can be learned, so the
model is meant to discover which time scales matter. The stated motivation is
that financial series carry structure at several horizons at once (intraday
speculation, weekly hedging, monthly allocation), and that multi-head attention
already provides parallel pathways that could specialise instead of duplicating.

The appeal is economy: other multi-scale time-series Transformers add branches,
patch sizes, or pyramids. This one adds `n_heads` scalars.

## 2. Input / output contract

```python
out = model(x_enc, x_dec, overlap_k=None, x_mem=None, attn_mask=None)
```

| Tensor | Shape | Role |
| --- | --- | --- |
| `x_enc` | `[B, T_enc, d_in]` | The lookback window. The only input that reaches the encoder. |
| `x_dec` | `[B, T_dec, d_in]` | **Interface only.** It is projected and the result is discarded. |
| `overlap_k` | `int` | **Unused.** Retained from an earlier quantile-forecasting path. |
| `x_mem` | `[B, T_mem, d_exog]` | Exogenous factors. Used only when `use_cross_attn=True`. |
| `attn_mask` | `[T_q, T_k]` | Optional. 1/True keeps, 0/False masks. |

Returns a dict, with `H = horizon`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `mu` | `[B, H]` | Conditional mean of the target (log return) per step. |
| `log_sigma` | `[B, H]` | Log conditional standard deviation, clamped to `[-20, 5]`. |
| `logits` | `[B, H]` | Direction logit per step. Apply the sigmoid outside the model. |

`d_exog` need not equal `d_in`: the exogenous projection is created lazily on
the first forward pass that supplies `x_mem`, sized to whatever arrives.

## 3. Rotary position embedding

Let `D_r` be the rotary dimension (defaults to the head dimension `D_h`) and
`m = D_r / 2`. The base frequencies form a geometric ladder:

```
inv_freq[i] = theta^(-i/m),            i = 0, 1, ..., m-1
```

For an integer position `t` the angles are `phi[t, i] = t * inv_freq[i]`, and
the tables are built by duplicating the half-width block:

```
cos[t] = concat( cos(phi[t]), cos(phi[t]) )     -> [T, D_r]
sin[t] = concat( sin(phi[t]), sin(phi[t]) )     -> [T, D_r]
```

Rotation uses the half-split convention (first half against second half):

```
rotate_half(x) = concat( -x[..., m:], x[..., :m] )
apply_rope(x)  = x * cos + rotate_half(x) * sin
```

RoPE is applied to the leading `rope_dim` channels of `q` and `k` only; any
remaining channels pass through unrotated. With the default
`rope_dim = d_head`, nothing passes through.

Note that **one head spans the whole geometric ladder**. With `theta = 10000`
and `m = 8` the wavelengths inside a single head run from about 6 steps to
about 20,000 steps. This matters for the redesign and is discussed in
[`03-known-issues.md`](03-known-issues.md) sections 2 and 6.

## 4. Per-head frequency scaling: the core mechanism

Each `MultiHeadAttentionRoPE` module owns a parameter vector `head_scaling` of
length `n_heads`, registered with `requires_grad = learnable_head_scaling`.
Writing `s_h = max(head_scaling[h], 1e-3)`, the intended angle for head `h` is

```
phi_h[t, i] = t * s_h * inv_freq[i]
```

so `s_h` dilates or compresses the positional axis seen by that head.

**How it is computed.** Rather than evaluating the expression above directly,
the implementation builds the integer-position tables once and then samples
them at the fractional positions `t * s_h` by linear interpolation:

```
p        = t * s_h
i0       = clamp(floor(p), 0, T_base - 2)
i1       = i0 + 1
w        = p - i0
cos_h[t] = cos[i0] * (1 - w) + cos[i1] * w          (and likewise for sin)
```

Query and key streams are sampled independently, so cross-attention between
sequences of different lengths is supported. The result is broadcast to
`[1, T, n_heads, D_r]` so it aligns with `q, k` of shape `[B, T, n_heads, D_h]`.

The table is only built to length `max(2, T_q, T_k)`. **This is the origin of
the most serious defect in the code**; see
[`03-known-issues.md`](03-known-issues.md) section 1.

## 5. Attention

```
q, k, v = linear projections of the inputs, reshaped to [B, T, n_heads, D_h]
q, k    = apply per-head scaled RoPE to the first rope_dim channels
scores  = einsum("bthd,bshd->bhts", q, k) / sqrt(D_h)
```

A causal mask is applied when `use_causal_mask=True` and `T_q == T_k` and no
explicit mask of the right shape was given. An external `attn_mask` is applied
on top. Then softmax, attention dropout, and

```
y = einsum("bhts,bshd->bthd", attn, v)  ->  reshape  ->  o_proj  ->  proj_dropout
```

Attention maps are retained on the module (before dropout as `last_attn_nodrop`,
after dropout as `last_attn`) whenever `save_attn` is true, which it is by
default. This is what the head-behaviour visualisations consume. It also means
the module holds a reference to a `[B, n_heads, T_q, T_k]` tensor at all times.
Detached, but worth knowing about for memory.

## 6. Encoder block

Self-attention, optional cross-attention, then a feed-forward network, each with
a residual connection:

```
y = self_attn( ln1(x_enc), x_enc, attn_mask )
x = x_enc + y
if use_cross_attn and x_mem is not None:
    x = x + cross_attn( ln_cross(x), x_mem )
x = x + ff( ln2(x) )
```

The feed-forward network is
`Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model) -> Dropout`.

Two things to notice. Cross-attention runs with `use_causal_mask=False`, which
is correct for attending to exogenous series. And the self-attention call
normalises only the query input while passing the raw `x_enc` as key and value,
an asymmetry relative to standard pre-norm blocks; see
[`03-known-issues.md`](03-known-issues.md) section 5.

## 7. Pooling and the output heads

```
h = encode(x_enc)                       # [B, T_enc, d_model]
z = h[:, -1, :]  or  h.mean(dim=1)      # pooling = "last" | "mean"
z = ReLU( head_hidden(z) )              # [B, d_model]

mu        = mu_head(z).view(B, H)
log_sigma = clamp( vol_head(z).view(B, H), -20, 5 )
logits    = dir_head(z).view(B, H)
```

All three heads read the same pooled vector `z`; they share every parameter up
to their final linear layer. With `pooling="last"` and a causal mask, `z` is
the representation of the final time step only, which is therefore the only
query position at which the multi-scale machinery is ever exercised.

## 8. Configuration

The constructor accepts either a `RopeFormerConfig` dataclass or flat keyword
arguments. The flat path exists so the original `main.py` could forward its
argparse namespace directly, and it renames several fields:

| Flat keyword | Maps to | Default |
| --- | --- | --- |
| `nhead` | `n_heads` | 8 |
| `enc_layers` | `num_layers` | 2 |
| `base_theta` | `rope_theta` | 10000.0 |
| `head_freq_scales` | `head_scaling_init` | all ones |
| `in_dim` | `d_in` | 1 |
| `dropout` | both `attn_dropout` and `ff_dropout` | 0.0 |

Remaining fields: `d_model`, `d_ff`, `horizon`, `rope_dim`, `use_cross_attn`,
`use_causal_mask`, `pooling` (`"last"` / `"mean"`), `learnable_head_scaling`,
`dir_num_classes`.

`head_scaling_init` must have length `n_heads` or construction raises.
Unrecognised keyword arguments are silently dropped, which is how several
configuration options end up having no effect. See
[`03-known-issues.md`](03-known-issues.md) sections 3 and 4.

## 9. Reference configuration

The settings reported as best-performing in the original experiments:

```yaml
d_model: 64
nhead: 4
enc_layers: 2
ff_mult: 2                              # see issue 3: this does NOT give d_ff=128
dropout: 0.0
base_theta: 10000.0
head_freq_scales: [0.25, 0.5, 1.0, 2.0]
learnable_head_scaling: true
pooling: last
use_cross_attn: true
dir_num_classes: 1
enc_len: 60
horizon: 5
```

Selected by a 576-point grid search. That selection does not survive scrutiny:
the spread of directional accuracy across the grid matched the binomial noise
floor of the test set, and every run was additionally affected by issues 1 and
3. Treat these numbers as a starting point, not as evidence.
