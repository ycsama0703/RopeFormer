"""Diagnostics for the RoPE positional spectrum used by RopeFormer.

Two checks:

1. SPECTRUM  -- lists the wavelength of every rotary frequency pair and flags
   the ones whose wavelength far exceeds the encoder window, i.e. the
   dimensions whose phase barely rotates and therefore carry almost no
   positional information.

2. SCALING   -- builds the per-head scaled cos/sin tables and verifies they
   stay inside [-1, 1]. Heads with a frequency scale s_h > 1 currently fail
   this check; see docs/03-known-issues.md.

Usage:
    python tools/check_rope_spectrum.py
    python tools/check_rope_spectrum.py --d-model 64 --n-heads 4 --theta 10000 --seq-len 60
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ropeformer.ropeformer import MultiHeadAttentionRoPE


def report_spectrum(d_head: int, theta: float, seq_len: int) -> None:
    half = d_head // 2
    print(f"\n[SPECTRUM] d_head={d_head}  theta={theta:g}  seq_len={seq_len}")
    print(f"{'pair':>4}  {'inv_freq':>12}  {'wavelength':>12}   status")
    dead = 0
    for i in range(half):
        inv_freq = 1.0 / (theta ** (i / half))
        wavelength = 2 * math.pi / inv_freq
        if wavelength > 2 * seq_len:
            status = "DEAD  (phase barely rotates over the window)"
            dead += 1
        elif wavelength < 2:
            status = "aliased (below Nyquist)"
        else:
            status = "informative"
        print(f"{i:>4}  {inv_freq:>12.3e}  {wavelength:>12.1f}   {status}")
    print(f"  -> {dead}/{half} frequency pairs carry ~no positional "
          f"information at seq_len={seq_len}")
    suggested = math.exp(math.log(seq_len / (2 * math.pi)) * half / max(half - 1, 1))
    print(f"  -> theta giving a longest wavelength of ~{seq_len} steps: {suggested:.1f}")


def report_scaling(d_model: int, n_heads: int, theta: float, seq_len: int,
                   scales) -> None:
    print(f"\n[SCALING] d_model={d_model}  n_heads={n_heads}  scales={list(scales)}")
    attn = MultiHeadAttentionRoPE(
        d_model=d_model,
        n_heads=n_heads,
        rope_theta=theta,
        head_freq_scaling=torch.tensor(scales, dtype=torch.float32),
    )
    cos_q, sin_q, _, _ = attn._make_scaled_rope(
        seq_len, seq_len, torch.device("cpu"), torch.float32
    )
    ok = True
    for h, s in enumerate(scales):
        c = cos_q[0, :, h, :]
        lo, hi = float(c.min()), float(c.max())
        bad = lo < -1.0001 or hi > 1.0001
        ok = ok and not bad
        flag = "  <-- OUT OF RANGE" if bad else ""
        print(f"  head {h}  s={s:<5} cos in [{lo:8.3f}, {hi:7.3f}]"
              f"  (expected [-1, 1]){flag}")
    print("  -> PASS" if ok else
          "  -> FAIL: see docs/03-known-issues.md (cache is built to length "
          "max(2, T_q, T_k), so pos*s_h overshoots it when s_h > 1)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--scales", type=str, default="0.25,0.5,1.0,2.0",
                   help="comma-separated per-head frequency scales")
    a = p.parse_args()

    scales = [float(x) for x in a.scales.split(",")]
    if len(scales) != a.n_heads:
        p.error(f"--scales has {len(scales)} values but --n-heads is {a.n_heads}")

    report_spectrum(a.d_model // a.n_heads, a.theta, a.seq_len)
    report_scaling(a.d_model, a.n_heads, a.theta, a.seq_len, scales)


if __name__ == "__main__":
    main()
