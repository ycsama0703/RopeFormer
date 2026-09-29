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
allocate monthly, and their activity superimposes on one series. Existing
approaches take one of two routes. Econometric models such as HAR impose the
horizons by hand - daily, weekly, monthly - which are conventions rather than
estimates, and cannot adapt when a market's relevant horizons differ or shift
between calm and crisis. Deep sequence models, by contrast, do not represent
horizon structure at all. Between the two, nobody estimates the horizons. We
ask whether a deep model can **learn** the time scales that matter, and report
them.

### 2. Data

Primary source: daily crude oil spot prices from FRED - `DCOILWTICO` (WTI) and
`DCOILBRENTEU` (Brent), free at fred.stlouisfed.org, from 1986. Target:
realised volatility at a 5-day horizon from a 60-day lookback, as the standard
deviation of daily log returns - a daily proxy, since no intraday data is used.
Since the method concerns learned time scales, we also evaluate on one or two
further public series whose horizons differ. Splits are chronological.

### 3. Method

**Architecture and motivation.** We use an encoder-only Transformer over the
lookback window, since self-attention lets any two days interact directly. A
Transformer is order-blind unless position is encoded, so we use rotary
position encoding (RoPE), which encodes position by rotating each query and key
vector through angles proportional to its index across a ladder of frequencies.
Two observations follow. Every head receives the *same* positional encoding, so
the parallel pathways of multi-head attention are never induced to specialise
by time scale. And models that do capture multiple resolutions, such as MTST
and Pathformer, buy them by adding architecture. This suggests taking
multi-resolution from parallelism already present.

**The proposed model.** Each head therefore receives its own learnable scalar
multiplying position before the rotary angle is formed, so each views the
series at its own time scale. RoPE is a natural carrier because its frequencies
are explicit: once the band is re-specified for a 60-day window and narrowed
per head, a head's time scale becomes a reportable number of days. Per-head
positional scales are not new - ALiBi assigns fixed slopes - but transferring
them to rotary encodings and learning them is. The model outputs a conditional
log-volatility per step.

**Baselines and evaluation.** HAR-RV; an identical Transformer with one shared
positional encoding, which ablates the mechanism; and LSTM, GRU and
iTransformer. We score with QLIKE and RMSE on log-variance over several seeds
and report the learned scales.

### 4. Expected difficulties and solutions

- **Low signal-to-noise.** Return direction is close to unpredictable, so we
  target volatility and report multi-seed intervals rather than a single best
  run.
- **Mis-specified positional encoding.** RoPE's default frequency range suits
  NLP context lengths, not a 60-day window; we tune the rotary base to the
  window.
- **Data irregularities.** WTI printed negative in April 2020, breaking log
  returns; we screen non-positive prices and hold out a recent period.

---

## Notes for the group (not part of the submission)

- Section 4 deliberately carries no measurements. The noise-floor comparison
  and the frequency-spectrum count come from the earlier individual oil_index
  work, not from this group project, so they do not belong in this submission.
  They remain useful internally: see `tools/check_rope_spectrum.py` and
  [`03-known-issues.md`](03-known-issues.md). Once we have reproduced anything
  equivalent as a group, it can be cited in the final report.
- Section 3 promises narrowed per-head frequency bands. That work is not done
  yet; see [`04-redesign-directions.md`](04-redesign-directions.md) direction
  B.
- ALiBi is named in one clause only, for space. The final report needs a proper
  treatment; see [`02-related-work.md`](02-related-work.md) section 1.
- Proprietary Platts-style assessments (`PPXDK00` and the other six series) are
  deliberately left out of the submission. Add them as an extension only if
  access is confirmed.
- Head collapse (nothing forces the learned scales to stay distinct) was cut
  from section 4 for space. It is still a real risk: add a diversity penalty on
  scale spacing and log the trajectories. Put it back if a bullet frees up.
- Verify the two FRED series IDs on the site before submitting.
- Finance relevance was dropped: the official handout lists only problem,
  method, data and difficulties. The TA's outline included it, so if a marker
  expects it, one sentence can be folded into section 1.
- If trimming is needed, cut from section 2. Do not cut the derivation
  paragraph in section 3 - it is what makes this read as a deep learning
  project rather than a finance one.
