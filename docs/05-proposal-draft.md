# Project Proposal (draft)

Budget: half a page to one page. Section structure follows the outline
circulated by the TA (the official handout lists items 1-4; item 5 is the TA's
addition).

**Placeholders marked `[TBD]` need a group member to fill them in.**

---

## Learning Time Scales: Per-Head Rotary Position Encoding for Crude Oil Volatility Forecasting

### 1. Problem and motivation

Volatility in crude oil markets is persistent, and that persistence runs at
several horizons at once: participants speculate intraday, hedge weekly and
allocate monthly, and their activity superimposes on one series.
Existing approaches take one of two routes. Econometric models such as
HAR impose the horizons by hand - daily, weekly, monthly - which are
conventions rather than estimates, and cannot adapt when a market's relevant
horizons differ or shift between calm and crisis. Deep sequence models, by
contrast, do not represent horizon structure at all. Between the two, nobody
estimates the horizons. We ask whether a deep model can **learn** the time
scales that matter, and report them.

### 2. Data

Primary source: daily crude oil spot prices from FRED - `DCOILWTICO` (WTI) and
`DCOILBRENTEU` (Brent), free at fred.stlouisfed.org, from 1986. Target:
realised volatility at a 5-day horizon from a 60-day lookback, as the standard
deviation of daily log returns - a daily proxy, since no intraday data is
used. Since the method concerns
learned time scales, we also evaluate on one or two further public series whose
horizons differ. Splits are chronological.

### 3. Method

**Architecture, and how we arrived at the modification.** We use an
encoder-only Transformer over the lookback window, since self-attention lets
any two days interact directly. A Transformer is order-blind unless position is
encoded, so we use rotary position encoding (RoPE), which represents position
by rotating each query and key vector through angles proportional to its index,
across a geometric ladder of frequencies. Two observations followed. Every head
receives the *same* positional encoding, so the parallel pathways of multi-head
attention are never induced to specialise by time scale. And models that do
capture multiple resolutions, such as MTST and Pathformer, buy them by adding
architecture: branches, patch sizes, pyramid levels. Together these suggested
taking multi-resolution from parallelism already present.

**The proposed model.** Each head receives its own learnable scalar multiplying
position before the rotary angle is formed, so each views the series at its own
time scale and the model learns which scales matter. RoPE is a natural carrier
because its frequencies are explicit: once the band is re-specified for a
60-day window and narrowed per head, "the time scale of head *h*" becomes a
reportable number of days. Per-head positional scales are not new - ALiBi
assigns fixed per-head slopes - but transferring them to rotary encodings and
learning them is. The model outputs a conditional log-volatility per step, with
direction as an auxiliary head.

**Baselines.** HAR-RV, fit on the same daily proxy; an otherwise identical
Transformer with one shared positional encoding, ablating the mechanism
directly; and LSTM, GRU and iTransformer for reference.

### 4. Expected difficulties and solutions

- **Low signal-to-noise on direction.** Directional accuracy across 576
  preliminary configurations spread by 0.0249 against a binomial noise floor of
  0.0239, so differences were unidentifiable. We target volatility instead and
  report multi-seed confidence intervals, not best-of-grid results.
- **Positional encoding inherited from NLP.** At a 60-day window, five of eight
  rotary frequency bands have wavelengths above 200 days and carry almost no
  positional information. We set the rotary base from the window length and
  check the spectrum.
- **Structural breaks and non-positive prices.** WTI printed negative in April
  2020, making log returns undefined, and four decades span several market
  regimes. We screen non-positive prices and hold out a recent period.

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
- Proprietary Platts-style assessments (`PPXDK00` and the other six series)
  are deliberately left out of the submission. Add them as an extension only
  if access is confirmed.
- Head collapse (nothing forces the learned scales to stay distinct) was cut
  from section 4 for space. It is still a real risk: add a diversity penalty
  on scale spacing and log the trajectories. Put it back if a bullet frees up.
- Verify the two FRED series IDs on the site before submitting.
- If trimming is needed, cut from section 2 or the last bullet of section 4.
  Do not cut the two numbers, and do not cut the derivation paragraph in
  section 3 - it is what makes this read as a deep learning project rather than
  a finance one.
