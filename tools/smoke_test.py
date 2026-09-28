"""Minimal forward-pass check: the method runs standalone, with no data pipeline.

Usage:
    python tools/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ropeformer import RopeFormer, list_methods

B, T_ENC, T_DEC, T_MEM = 4, 60, 8, 60
D_IN, D_EXOG, HORIZON = 8, 12, 5


def main() -> None:
    print("registered methods:", list_methods())

    model = RopeFormer(
        d_model=64,
        nhead=4,
        enc_layers=2,
        d_ff=128,
        dropout=0.0,
        horizon=HORIZON,
        base_theta=10000.0,
        head_freq_scales=[0.25, 0.5, 1.0, 2.0],
        pooling="last",
        use_cross_attn=True,
        dir_num_classes=1,
        learnable_head_scaling=True,
        in_dim=D_IN,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    x_enc = torch.randn(B, T_ENC, D_IN)
    x_dec = torch.randn(B, T_DEC, D_IN)
    x_mem = torch.randn(B, T_MEM, D_EXOG)

    model.eval()
    with torch.no_grad():
        out = model(x_enc, x_dec, x_mem=x_mem)

    print("\noutputs:")
    for k, v in out.items():
        print(f"  {k:<10} {tuple(v.shape)}")

    expected = {"mu": (B, HORIZON), "log_sigma": (B, HORIZON), "logits": (B, HORIZON)}
    for k, shape in expected.items():
        assert k in out, f"missing output key: {k}"
        assert tuple(out[k].shape) == shape, f"{k}: {tuple(out[k].shape)} != {shape}"
    assert torch.isfinite(out["mu"]).all(), "mu contains non-finite values"

    print("\nlearned per-head RoPE scales (block 0):")
    scales = model.blocks[0].self_attn.head_scaling.detach()
    for h, s in enumerate(scales.reshape(-1).tolist()):
        print(f"  head {h}: s = {s:.4f}")

    print("\nOK")


if __name__ == "__main__":
    main()
