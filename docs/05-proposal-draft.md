# Project Proposal (draft)

Budget: half a page to one page. Current draft is ~410 words. Section lengths
follow the structure circulated by the TA (the official handout lists items
1-4; item 5 is the TA's addition).

**Placeholders marked `[TBD]` need a group member to fill them in.**

---

## Learning Time Scales: Per-Head Rotary Position Encoding for Crude Oil Volatility Forecasting

### 1. Problem and motivation

Crude oil volatility is driven by participants acting at different horizons:
intraday speculation, weekly hedging, monthly allocation. Its structure is
therefore inherently multi-scale. Existing approaches either fix those horizons
by hand or do not represent horizon structure at all. We ask whether a model
can instead **learn** the time scales that matter, and report them.

### 2. Data

Daily crude oil price assessment series: 4,129 trading days from 2009-08-14 to
2025-09-12, covering seven contract series (spot and forward grades), together
with engineered volatility and calendar factors. Source: `[TBD - name the
provider and add a URL if the series are public]`. The prediction target is
realised volatility of log returns over a 5-day horizon, from a 60-day lookback
window. Splits are chronological: train to 2021-03-31, validate to 2023-03-31,
test thereafter.

### 3. Method

Our base model is an encoder-only Transformer over the lookback window. The
modification we study is to give **each attention head its own learnable time
scale**, applied by scaling the frequency of its rotary position encoding.
Multi-resolution then comes from parallelism that multi-head attention already
provides, rather than from added branches or patch sizes as in MTST and
Pathformer. We additionally re-specify the rotary frequency band for a 60-day
window and narrow each head's band, so that "the time scale of head *h*"
becomes a well-defined quantity in days that can be reported. The model emits a
conditional log-volatility per step, with direction as an auxiliary head.

Baselines: HAR-RV, which fixes daily, weekly and monthly horizons by
convention; an otherwise identical Transformer with a single shared positional
encoding, which ablates the mechanism directly; and LSTM, GRU and iTransformer
for reference. The design isolates one mechanism, and its learned parameters
are directly interpretable.

### 4. Expected difficulties and solutions

- **Low signal-to-noise on direction.** In preliminary work, directional
  accuracy across 576 configurations had a spread of 0.0249 against a binomial
  noise floor of 0.0239, so accuracy differences were unidentifiable. We
  therefore target volatility, which is far more forecastable, and report
  multi-seed confidence intervals rather than best-of-grid results.
- **Positional encoding inherited from NLP.** At a 60-day window, five of eight
  rotary frequency bands have wavelengths above 200 days and carry almost no
  positional information. We set the rotary base from the window length and
  verify the spectrum with a diagnostic script.
- **Head collapse.** Nothing forces the learned scales to stay distinct. We add
  a diversity penalty on their spacing and log their trajectories during
  training.
- **Limited data.** Roughly 2,900 training days. We keep the model small, use
  strictly chronological splits, and stop early on validation loss.

### 5. Finance relevance

Volatility forecasts drive margining, hedging and risk limits. The learned
per-head scales additionally give a direct readout of the horizon at which a
market is currently organised, which HAR fixes by convention rather than
estimating.

---

## Notes for the group (not part of the submission)

- The two numbers quoted in section 4 are reproducible: the noise-floor
  comparison comes from the original 576-point grid search, and the frequency
  spectrum from `tools/check_rope_spectrum.py`. Most groups write generic
  difficulties, so keeping these specific is the cheapest way to stand out.
- Section 3 claims we will narrow the per-head frequency bands. That work is
  not done yet; see [`04-redesign-directions.md`](04-redesign-directions.md)
  direction B.
- Section 3 does not mention ALiBi, for space. The final report must cite it;
  see [`02-related-work.md`](02-related-work.md) section 1.
- Word counts if sections need trimming: 1) 55, 2) 70, 3) 165, 4) 150, 5) 40.
