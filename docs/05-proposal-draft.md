# Project Proposal (draft)

Budget: half a page to one page. Section structure follows the outline
circulated by the TA (the official handout lists items 1-4; item 5 is the TA's
addition).

**Placeholders marked `[TBD]` need a group member to fill them in.**

---

## Learning Time Scales: Per-Head Rotary Position Encoding for Crude Oil Volatility Forecasting

### 1. Problem and motivation

Crude oil volatility is driven by participants acting at different horizons -
intraday speculation, weekly hedging, monthly allocation - so its structure is
inherently multi-scale. Existing approaches either fix those horizons by hand,
as HAR does with daily, weekly and monthly components, or do not represent
horizon structure at all. We ask whether a deep model can instead **learn** the
time scales that matter.

### 2. Data

Daily crude oil price assessments: 4,129 trading days from 2009-08-14 to
2025-09-12, seven contract series plus volatility and calendar factors. Source:
`[TBD - provider, and a URL if public]`. Target: realised volatility of log
returns at a 5-day horizon from a 60-day lookback. Chronological splits: train
to 2021-03-31, validate to 2023-03-31, test thereafter.

### 3. Method

**Base architecture.** An encoder-only Transformer over the lookback window.
Self-attention lets any two days interact directly, which suits a series whose
relevant lag is unknown and may itself change.

**How we arrived at the modification.** A Transformer is order-blind unless
position is encoded, so we use rotary position encoding (RoPE), which
represents position by rotating each query and key vector through angles
proportional to its index, across a geometric ladder of frequencies. Two
observations followed. Every head receives the *same* positional encoding, so
although multi-head attention supplies parallel pathways, nothing induces any
of them to specialise by time scale. And models that do capture multiple
resolutions, such as MTST and Pathformer, obtain them by adding architecture:
branches, patch sizes, pyramid levels. Together these suggested taking
multi-resolution from parallelism already present rather than adding more.

**The proposed model.** Each head receives its own learnable scalar multiplying
position before the rotary angle is formed, so each views the series at its own
time scale and the model learns which scales matter. RoPE is a natural carrier
because its frequencies are explicit: once the band is re-specified for a
60-day window and narrowed per head, "the time scale of head *h*" becomes a
reportable number of days. Per-head positional scales are not new in themselves
- ALiBi assigns fixed per-head slopes - but transferring the idea to rotary
encodings and learning the scales is. The model outputs a conditional
log-volatility per step, with direction as an auxiliary head.

**Baselines.** HAR-RV, which fixes the horizons by convention; an otherwise
identical Transformer with one shared positional encoding, ablating the
mechanism directly; and LSTM, GRU and iTransformer for reference.

### 4. Expected difficulties and solutions

- **Low signal-to-noise on direction.** In preliminary work, directional
  accuracy across 576 configurations had a spread of 0.0249 against a binomial
  noise floor of 0.0239, so differences were unidentifiable. We therefore
  target volatility, which is far more forecastable, and report multi-seed
  confidence intervals rather than best-of-grid results.
- **Positional encoding inherited from NLP.** At a 60-day window, five of eight
  rotary frequency bands have wavelengths above 200 days and carry almost no
  positional information. We set the rotary base from the window length and
  verify the spectrum with a diagnostic script.
- **Head collapse.** Nothing forces the learned scales to stay distinct. We add
  a diversity penalty on their spacing and log their trajectories.
- **Limited data.** Roughly 2,900 training days. We keep the model small, use
  strictly chronological splits, and stop early on validation loss.

### 5. Finance relevance

Volatility forecasts drive margining, hedging and risk limits. The learned
per-head scales additionally read out the horizon at which a market is
currently organised, which HAR fixes by convention rather than estimating.

---

## Notes for the group (not part of the submission)

- Both numbers in section 4 are reproducible: the noise-floor comparison from
  the original 576-point grid search, the frequency spectrum from
  `tools/check_rope_spectrum.py`. Most groups write generic difficulties, so
  keeping these specific is the cheapest way to stand out.
- Section 3 promises narrowed per-head frequency bands. That work is not done
  yet; see [`04-redesign-directions.md`](04-redesign-directions.md) direction B.
- ALiBi is named in one clause only, for space. The final report needs a proper
  treatment; see [`02-related-work.md`](02-related-work.md) section 1.
- If trimming is needed, cut from section 2 or the last bullet of section 4.
  Do not cut the two numbers, and do not cut the derivation paragraph in
  section 3 - it is what makes this read as a deep learning project rather than
  a finance one.
