# Redesign Directions

Candidate directions for reworking the method, with the reasoning behind each.
Nothing here is decided. This document exists so the group can argue about the
options with the same information, and so the method section of the proposal
can be written from a position rather than a list.

## 0. Two constraints to accept first

**The mechanism has never actually run.** Issues 1 and 3 in
[`03-known-issues.md`](03-known-issues.md) were active in every previous
experiment: one head always had a corrupted positional encoding, and every
feed-forward network was a width-2 bottleneck. No existing result tells us
whether per-head scaling helps or hurts.

**The previous evaluation target cannot discriminate.** In the original grid
search, directional accuracy on the test set had a standard deviation across
576 configurations of 0.0249. The binomial noise floor for that test set is
`sqrt(0.25/436) = 0.0239`. The two are the same number: the entire spread across
the hyper-parameter grid was indistinguishable from chance, and the best
configuration was no better than what picking the maximum of 576 coin-flip
experiments would give. Overlapping windows at horizon 5 make the effective
sample size smaller still.

Daily direction of an oil index is close to unpredictable; this is a property of
the data, not a failure of the model. Volatility is different, and is one of the
most robustly forecastable quantities in finance. **Any redesign judged on
directional accuracy cannot be shown to work, because the referee is blind.**
This should drive the choice of target as much as the choice of mechanism.

---

## A. Measure before imposing

*Cheap, and it decides whether the rest is worth doing.*

The premise is that attention heads need to be pushed into different time
scales. Nobody has checked whether they do it on their own. Multi-head
attention already learns whatever each head needs, so forcing scale
specialisation may be a constraint rather than a gift.

The experiment is to train a plain Transformer, then measure the effective time
scale of each head: the centre of mass of attention over distance, the
effective receptive field, or the decay length of attention weight against lag.

- If heads already separate by scale, the premise of the method is empty, **and
  that is a result worth reporting.**
- If heads cluster, we have a real justification for adding the inductive bias,
  and a measured baseline to improve on.

Without this, the motivation for the whole method is an assumption. This should
come before every other direction.

---

## B. Narrow the band: heads as a learned filter bank

*Repairs the conceptual flaw in section 6 of the issues document.*

Give each head a narrow, learnable frequency band instead of one scalar on top
of the full geometric ladder. For example, parameterise a centre frequency and
a width:

```
inv_freq[h] = f_centre[h] * geomspace(1/sqrt(w[h]), sqrt(w[h]))
```

Two parameters per head. Consequences:

- "The time scale of head `h`" becomes `2 * pi / f_centre[h]`, a real number in
  days that can be plotted and reported.
- Each head becomes a band-pass filter over time rather than a broadband
  encoder. This is something ALiBi structurally cannot express, so it is where
  our differentiation from the closest prior work actually lives.
- Constraining wavelengths to `[2, T]` keeps the learned values interpretable
  and out of the dead region described in issue 2.

This is the smallest change that makes the method's central claim literally
true instead of loosely true. It should be paired with a diversity penalty on
the spacing of the centres, or the heads may simply collapse onto each other
(issue 7).

---

## C. Make the time scale an output, not a parameter

*The most interesting reframing, and the only one that escapes the blind referee.*

Instead of "the model has multi-scale machinery so that it forecasts better",
make the model's job to estimate **the dominant time scale of the market at
each point in time**. The deliverable is a curve `tau(t)`, not a return
forecast.

It can then be validated against measurable properties of the data rather than
against an unpredictable target: the autocorrelation decay length of the local
window, volatility persistence, the steepness of the term structure, known
crisis dates. Forecasting becomes an auxiliary task whose only role is to force
`tau` to mean something.

Why this matters: we have established that direction is unpredictable, so any
method judged on it cannot be shown to work. A method whose output is "the
current dominant horizon is about N days" is validated against things we can
actually measure. **The project then has a result even if predictive accuracy
does not improve at all**, which given the data is a realistic outcome to plan
for.

This changes the model from a predictor into an instrument. As far as we know,
reading learned RoPE scales as an economic quantity has not been done.

---

## D. Put the scale in the aggregation operator instead of the position code

*The cleanest mechanical reconstruction.*

A positional encoding is descriptive: it says how far apart two steps are.
"Viewing the series at scale `tau`" is an operation: a smoothing, a filtering, an
aggregation. Encoding it in the position code means implementing an operation
with a description, which is why the current mechanism fights RoPE's broadband
design.

The alternative is to give each head's **attention kernel** an intrinsic scale,
so head `h` aggregates at bandwidth `tau_h` and attention decides which scale
matters for the current prediction. The multi-resolution decomposition is then
built explicitly rather than coaxed out of the position encoding, and mechanism
and story agree.

---

## E. Attach the scale to variables rather than heads

*Uses data we already have.*

The dataset contains seven price series of different grades and tenors. They do
not share a characteristic time scale: spot reacts quickly, forward contracts
slowly, and information propagates between them with a lag.

So learn a **per-variable** time scale rather than a per-head one, and read off
the scale hierarchy and lead-lag structure across the oil complex. "Which part
of the complex moves first, and at what horizon" is a real question with a
checkable answer. This crosses iTransformer's variables-as-tokens formulation
with the scale idea; both pieces are already available in the original
repository.

---

## F. The unexciting comparator

Decompose the input explicitly first (wavelet transform, or simple
multi-horizon differences), then run ordinary attention on the decomposed
representation. This is the MTST family. It is the least elegant option and
probably the strongest numerically. We should know it is there, implement it if
time allows, and not pretend it does not exist.

---

## Recommendation

**A first**, because it determines whether any of the others are worth doing.

Then **C**, because it is the only direction that produces a result independent
of predictive accuracy, which the data tells us we should not count on.

**B** is the natural mechanical companion to C: narrowing the bands is what
makes the scale readable in the first place, so B and C reinforce each other.

**D** and **E** are good reconstructions but both still compete on accuracy,
which is the axis we have the least reason to expect to win on.

Whatever we choose, the repairs in issues 1 to 3 come first. They are cheap,
and no measurement means anything until they are done.
