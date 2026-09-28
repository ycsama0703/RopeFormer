# Related Work and Positioning

Purpose of this document: make sure we position the method honestly in the
proposal, and know which baseline any reviewer will compare us against.

> Bibliographic details below are from memory. **Verify every citation against
> the actual paper before it goes into a submitted document.**

## 1. The closest prior work: ALiBi

Press et al., *Train Short, Test Long: Attention with Linear Biases Enables
Input Length Extrapolation*, ICLR 2022 (arXiv:2108.12409).

ALiBi adds a distance penalty `-m_h * |i - j|` to the attention scores, and
**gives each head a different slope `m_h`, drawn from a geometric sequence**
(1/2, 1/4, 1/8, ...). Different heads therefore decay over different distances,
which is to say: each head operates at its own positional length scale.

That is the same idea as ours. Our `head_freq_scales = [0.25, 0.5, 1.0, 2.0]`
is likewise a geometric sequence assigning one positional scale per head. The
differences are that we act on the rotary angle rather than on an additive
bias, and that we let the coefficients be learned.

**We must cite this.** Not for correctness reasons but for tactical ones: if a
reader knows ALiBi and we have not mentioned it, the two available readings are
that we did not read the literature or that we chose not to mention it. Both are
worse than the honest framing, which is a perfectly respectable one:

> We transfer ALiBi's per-head positional-scale idea to rotary embeddings, make
> the scales learnable, and apply the result to joint direction and volatility
> forecasting of crude oil indices.

Note also the asymmetry that works in our favour: an ALiBi head can only decay
monotonically with distance, so it can express "near" versus "far" but not a
band. A rotary head with a narrow frequency band can express band-pass
behaviour, i.e. sensitivity to a specific periodicity. That is a real capability
difference and it is the strongest argument for using RoPE as the carrier, but
only once the bands are actually narrowed. See
[`03-known-issues.md`](03-known-issues.md) section 6.

## 2. Rotary embeddings and their rescaling

- **RoPE** — Su et al., *RoFormer*, arXiv:2104.09864. The rotary encoding we
  build on.
- **Position Interpolation** — Chen et al., arXiv:2306.15595. Divides positions
  by a factor so a model trained at one context length works at a longer one.
- **NTK-aware scaling** and **YaRN** — Peng et al., arXiv:2309.00071. YaRN
  notably does *not* apply a uniform factor: it treats different frequency bands
  differently depending on how their wavelength compares with the context
  length.

The arithmetic of scaling positions or frequencies in RoPE is therefore well
established. The motivation differs from ours: that line of work rescales in
order to extrapolate to longer contexts, with a single policy shared by all
heads. Ours varies the scale *across heads* in order to induce specialisation.
The novelty is in the purpose, not in the operation, and the write-up should
say so rather than imply otherwise.

YaRN's per-band treatment is also a useful precedent for the redesign direction
in section 6 of the issues document: differentiating frequency bands within
RoPE is an established move.

## 3. Multi-scale time-series Transformers

This is the family we compete with. All of them obtain multiple resolutions by
**adding architecture**:

- **MTST**, *Multi-resolution Time-Series Transformer* (AISTATS 2024) — parallel
  branches with different patch sizes. The closest competitor in spirit, and
  already in the project's reading list.
- **Pathformer** (ICLR 2024) — multi-scale division with adaptive pathway
  selection.
- **Scaleformer** (ICLR 2023) — iterative refinement across resolutions.
- **Pyraformer** (ICLR 2022) — a pyramidal attention tree over time.

Our contrast is that we add `n_heads` scalars instead of branches, reusing the
parallelism multi-head attention already has. This is the most attractive part
of the story and it should be stated as an efficiency and parsimony argument,
not as a performance claim, unless we can actually demonstrate performance.

For general context in the same literature: Informer (AAAI 2021), Autoformer
(NeurIPS 2021), FEDformer (ICML 2022), PatchTST (ICLR 2023), and iTransformer
(ICLR 2024). The last of these is already implemented in the original
experiment repository and is a natural baseline.

## 4. The finance side: multi-horizon volatility

Corsi, *A Simple Approximate Long-Memory Model of Realized Volatility*, Journal
of Financial Econometrics, 2009. The HAR-RV model regresses realised volatility
on daily, weekly, and monthly components. It is the standard workhorse for
volatility forecasting, and its inductive bias is explicitly multi-scale: it
assumes market participants operate at distinct horizons and that their
activity superimposes.

This matters to us for two reasons. It supplies an economic motivation for
per-head time scales that is stronger than "different heads should do different
things" — our learned scales can be presented as a learned, attention-based
relaxation of HAR's hand-fixed daily/weekly/monthly decomposition. And it
supplies a strong, cheap, universally recognised baseline for the volatility
target.

It also suggests a free interpretability device: initialise the head scales at
horizons of 1, 5, and 22 days and report where they move to.

## 5. Summary of the honest claim

What is genuinely ours:

1. Applying per-head positional scaling to rotary embeddings, with the scales
   learned rather than fixed.
2. Combining it with a joint direction and volatility output for a financial
   target.
3. (If we pursue the redesign) Narrowing each head to a frequency band, which
   turns the heads into a learned filter bank over time and produces a
   reportable time scale per head. This is the part with the least prior art.

What is not ours: the idea that heads should have different positional length
scales (ALiBi), the arithmetic of rescaling RoPE (Position Interpolation, YaRN),
and multi-resolution time-series modelling in general (MTST and its family).

For a course project this is more than sufficient. Novelty is not the grading
criterion; a clear question, a controlled experiment, and an honest reading of
the result are.
