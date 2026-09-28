# Known Issues

Everything in sections 1 to 5 was reproduced against the code in this
repository. Sections 6 and 7 are not bugs; they are design problems that the
redesign has to answer.

**Read this before trusting any earlier experimental result.** Issues 1 and 3
were both active in every run of the original 576-point grid search, so those
numbers describe a model that was crippled in two independent ways.

| # | Issue | Severity | Affected past runs? |
| --- | --- | --- | --- |
| 1 | RoPE table overrun when `s_h > 1` | Critical | Yes, all |
| 2 | `theta=10000` wastes most of the spectrum at `T=60` | High | Yes, all |
| 3 | `ff_mult` collapses the FFN to width 2 | Critical | Yes, all |
| 4 | `dir_num_classes` and `task_mode` silently ignored | Medium | Yes (features unreachable) |
| 5 | Pre-norm applied to the query only | Low | Yes, all |
| 6 | A RoPE head has no single characteristic scale | Conceptual | n/a |
| 7 | Nothing keeps the head scales apart | Conceptual | n/a |

---

## 1. RoPE table overrun when `s_h > 1` (critical)

**Symptom.** Cosine and sine values that must lie in `[-1, 1]` reach roughly
`-53`. The positional encoding of any head with a scale above 1 is garbage over
the later part of the sequence.

**Reproduce.**

```bash
python tools/check_rope_spectrum.py
```

```
head 0  s=0.25  cos in [  -1.000,   1.000]
head 1  s=0.5   cos in [  -1.000,   1.000]
head 2  s=1.0   cos in [  -1.000,   1.000]
head 3  s=2.0   cos in [ -53.296,   7.357]   <-- OUT OF RANGE
```

**Root cause.** In `_make_scaled_rope` the cos/sin table is built only to length
`max(2, T_q, T_k)`, i.e. exactly the sequence length. Positions are then sampled
at `p = t * s_h`. When `s_h > 1`, `p` runs past the end of the table. The index
is clamped:

```python
idx0 = torch.floor(pos_scaled).to(torch.long).clamp_(0, Tbase - 2)
idx1 = idx0 + 1
w    = (pos_scaled - idx0.to(pos_scaled.dtype))...   # never clamped
```

but the interpolation weight `w` is not. With `T=60` and `s_h=2.0`, position 59
gives `p=118`, `idx0=58`, and `w=60`. The final line is then not an
interpolation but a linear extrapolation with weight 60:

```
cos = cos[58] * (1 - 60) + cos[59] * 60
```

**Impact.** `run_grid.py` used `FREQ_MAP = {2: [0.5, 1.5], 4: [0.25, 0.5, 1.0, 2.0]}`.
Both entries contain a scale above 1, so **every one of the 576 grid runs had at
least one head with a corrupted positional encoding** over roughly the second
half of every window. Because `learnable_head_scaling=True`, any head that
drifts above 1.0 during training falls into this as well.

**Fix.** Do not interpolate. The tables are analytic, so evaluate the angle
directly at the scaled position:

```python
phi = t[:, None] * s_h * inv_freq[None, :]    # exact, no table, no error
cos, sin = torch.cos(phi), torch.sin(phi)
```

The build-a-table-and-interpolate pattern comes from LLM long-context RoPE
scaling code, where the point is to extrapolate beyond the trained context
length. Here the window length is fixed and known, so it buys nothing and
costs correctness.

---

## 2. `theta = 10000` wastes most of the spectrum at `T = 60` (high)

**Symptom.** Five of the eight rotary frequency pairs encode almost no
positional information.

**Reproduce.** The `SPECTRUM` section of `tools/check_rope_spectrum.py`:

```
pair      inv_freq    wavelength   status
   0     1.000e+00           6.3   informative
   1     3.162e-01          19.9   informative
   2     1.000e-01          62.8   informative
   3     3.162e-02         198.7   DEAD
   4     1.000e-02         628.3   DEAD
   5     3.162e-03        1986.9   DEAD
   6     1.000e-03        6283.2   DEAD
   7     3.162e-04       19869.2   DEAD
  -> 5/8 frequency pairs carry ~no positional information at seq_len=60
  -> theta giving a longest wavelength of ~60 steps: 13.2
```

**Root cause.** `theta = 10000` is inherited from language models with context
windows of thousands of tokens. Our encoder window is 60 steps. A frequency
whose wavelength is 20,000 steps rotates by less than one percent of a cycle
across the whole window, so those channels are effectively constant and carry
no positional signal.

**Impact.** Roughly 62% of the rotary dimensions do nothing. It also weakens the
core mechanism: the per-head scales span a factor of 8, but the frequency ladder
inside each head spans about 3.5 orders of magnitude, so `s_h` is a small
perturbation on a band that already covers everything. See section 6.

**Fix.** Choose `theta` from the window length rather than from LLM convention.
Setting the longest wavelength to about `T` gives `theta` near 13 for
`d_head=16, T=60`. This is a one-number change and it makes the per-head scales
meaningful relative to the within-head spread.

---

## 3. `ff_mult` collapses the feed-forward network to width 2 (critical)

**Symptom.** With the reference configuration, every block's feed-forward
network is a 2-dimensional bottleneck instead of the intended 128.

**Reproduce.**

```python
import sys; sys.path.insert(0, "src")
from ropeformer import RopeFormer
m = RopeFormer(d_model=64, nhead=4, enc_layers=2, ff_mult=2, horizon=5, in_dim=8,
               head_freq_scales=[0.25, 0.5, 1.0, 2.0])
print(m.cfg.d_ff)           # 2
print(m.blocks[0].ff.net[0])  # Linear(in_features=64, out_features=2)
print(m.blocks[0].ff.net[3])  # Linear(in_features=2,  out_features=64)
```

**Root cause.** Two places disagree about whether `ff_mult` is a multiplier or
an absolute width. In the constructor:

```python
d_ff = kwargs.pop("d_ff", kwargs.pop("ff_mult", 2)) \
       if isinstance(kwargs.get("ff_mult", None), int) \
       else kwargs.pop("d_ff", 512)
```

The test is inverted. When `ff_mult` is an `int` it takes the branch that
treats the value as an absolute width, so `ff_mult: 2` yields `d_ff = 2`.
The block constructor then repeats the same test:

```python
d_ff = self.cfg.d_ff if isinstance(self.cfg.d_ff, int) \
       else int(self.cfg.d_model * float(self.cfg.d_ff))
```

and again sees an `int`, so it uses 2 literally. The multiplier path is only
reachable by passing a `float`, which `ff_mult: 2` in YAML does not produce.

**Impact.** Every feed-forward network in every past run had a hidden width of
2. The FFN is where a Transformer block does most of its per-token computation,
so the model was running at a small fraction of its intended capacity
throughout the grid search.

**Workaround.** Pass `d_ff` explicitly as an integer and never pass `ff_mult`.

**Fix.** Pick one convention. The clearest is to drop `ff_mult` entirely and
require an explicit `d_ff`.

---

## 4. `dir_num_classes` and `task_mode` are silently ignored (medium)

**Symptom.** A three-class direction head cannot be built, and `task_mode:
dist` has no effect.

**Reproduce.**

```python
m = RopeFormer(d_model=64, nhead=4, enc_layers=2, d_ff=128, horizon=5, in_dim=8,
               head_freq_scales=[0.25, 0.5, 1.0, 2.0],
               dir_num_classes=3, task_mode="dist")
print(m.cfg.dir_num_classes)   # 3   parsed into the config
print(m.dir_num_classes)       # 1   what is actually used
print(hasattr(m, "task_mode")) # False
```

**Root cause.** Both values are consumed by `kwargs.pop(...)` while building the
config, and are then re-read from `kwargs` after they are already gone:

```python
self.dir_num_classes = int(getattr(self, "dir_num_classes",
                                   kwargs.get("dir_num_classes", 1)))
```

`self.dir_num_classes` does not exist yet, so `getattr` falls through to
`kwargs.get`, which now returns the default of 1. `task_mode` is never assigned
at all, so the `dist` branch in `forward` reads
`getattr(self, "task_mode", "vol+dir")` and is unreachable dead code.

The same pattern hits `self.pooling`, which is always `"last"`. That one is
harmless only because `pool()` reads `self.cfg.pooling` instead.

**Impact.** Only binary direction is reachable, and the distribution-only mode
cannot be selected. No past result is wrong because of this, but two documented
options never existed.

**Fix.** Read these from `self.cfg` after it is built, not from the already
drained `kwargs`.

---

## 5. Pre-norm is applied to the query only (low)

In `EncoderBlock.forward`:

```python
y = self.self_attn(self.ln1(x_enc), x_enc, attn_mask=attn_mask)
```

The query input is layer-normalised, the key and value inputs are not. Standard
pre-norm self-attention normalises all three. The residual stream is otherwise
correct, and this is not obviously fatal, but it is an asymmetry that was very
likely unintended and it makes the block harder to compare against a reference
implementation.

---

## 6. A RoPE head has no single characteristic scale (conceptual)

This is the deepest problem, and it is about the idea rather than the code.

The premise is "head `h` operates at time scale `s_h`". But RoPE is built so
that a single head already contains a whole geometric ladder of frequencies:
that is the point of the design. With `theta=10000` and `m=8`, one head
simultaneously carries wavelengths from about 6 steps to about 20,000 steps.
A RoPE head is inherently broadband.

So multiplying position by one scalar does not give a head a scale. It slides
that head's already very wide spectrum by a modest amount. The quantity the
method tries to control is not a well-defined property of the representation it
is built on.

Compare ALiBi, where the same per-head idea is coherent: an ALiBi head has
exactly one slope, therefore exactly one decay length, so per-head slopes
genuinely partition the range of scales. Porting the idea from ALiBi to RoPE
loses the property that made it work.

**Implication for the redesign.** To make "head = scale" literally true, each
head needs a narrow frequency band rather than the full ladder: learn a centre
frequency and a bandwidth per head. Then `2 * pi / f_centre_h` is a real
number of days that can be reported, and each head becomes a band-pass filter
over time, which is something ALiBi cannot express. See
[`04-redesign-directions.md`](04-redesign-directions.md).

---

## 7. Nothing keeps the head scales apart (conceptual)

With `learnable_head_scaling=True` the scales are free parameters under no
constraint. Nothing in the loss rewards the heads for covering different scales,
so they may all converge to the same value, which would quietly remove the
mechanism the method is named after.

We do not know whether this happens, because the learned scales were never
logged. Whether the heads separate or collapse is a cheap and genuinely
informative thing to measure, and it should be checked before any claim about
multi-scale behaviour is made. Candidate remedies are a diversity penalty on
the spacing of the scales, or an orthogonality-style constraint on the bands.
