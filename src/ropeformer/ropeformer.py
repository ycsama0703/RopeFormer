# ropeformer.py
# RopeFormer → Volatility + Direction (two-head) version
# - Rotary Positional Embedding (RoPE) with per-head frequency scaling
# - Encoder-only stack (Self-Attn) + optional Cross-Attn
# - Outputs:
#     {"logits": [B, H], "log_sigma": [B, H]}
#   where H=horizon (multi-step)
#
# Author: you + ChatGPT

from dataclasses import dataclass
from typing import Optional, Literal, Dict
from .registry import register_method
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: RoPE
# -----------------------------
def build_rotary_cache(seq_len: int, dim: int, theta: float = 10000.0, device=None, dtype=None):
    """
    Build standard RoPE cos/sin cache for length seq_len and rotary dim=dim (must be even).
    Returns cos, sin shaped [seq_len, dim]
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    half_dim = dim // 2
    # base frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, half_dim]
    # interleave cos/sin
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [seq_len, dim]
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor):
    # split last dim into two halves and rotate: (x1, x2) -> (-x2, x1)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: [B, T, H, Dh]   cos/sin: [T, Dh]  (broadcast over batch and heads)
    """
    # expand cos/sin to [1, T, 1, Dh]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


# -----------------------------
# Multi-Head Attention with RoPE
# -----------------------------
class MultiHeadAttentionRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_dim: Optional[int] = None,
        # per-head RoPE scaling
        head_freq_scaling: Optional[torch.Tensor] = None,  # [n_heads], each > 0
        learnable_head_scaling: bool = False,
        use_bias: bool = True,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_causal_mask = use_causal_mask

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # RoPE config
        self.rope_theta = rope_theta
        self.rope_dim = rope_dim if rope_dim is not None else self.d_head  # usually Dh
        if self.rope_dim > self.d_head:
            raise ValueError("rope_dim cannot exceed d_head")

        # per-head scaling factors (s_h)
        if head_freq_scaling is None:
            init = torch.ones(n_heads)
        else:
            assert head_freq_scaling.shape == (n_heads,)
            init = head_freq_scaling
        self.head_scaling = nn.Parameter(init, requires_grad=learnable_head_scaling)

        self._cache_cos: Optional[torch.Tensor] = None  # [T, rope_dim]
        self._cache_sin: Optional[torch.Tensor] = None

        self.save_attn = True          # 开关：是否保存注意力
        self.last_attn = None          # 缓存最近一次注意力 [B,H,Tq,Tk]
        self.last_attn_nodrop = None    # dropout 之前（标准注意力分布）


    def _ensure_rope_cache(self, T: int, device, dtype):
        if (
            self._cache_cos is None
            or self._cache_cos.shape[0] < T
            or self._cache_cos.device != device
            or self._cache_cos.dtype != dtype
        ):
            cos, sin = build_rotary_cache(T, self.rope_dim, self.rope_theta, device=device, dtype=dtype)
            self._cache_cos = cos
            self._cache_sin = sin

    def _make_scaled_rope(self, T_q: int, T_k: int, device, dtype):
        """
        Build per-head scaled cos/sin for Q and K streams.
        - 允许 base_cos/base_sin 为 [Tbase, Drot] 或 [Tbase,1,Drot] 或 [Tbase,H,1,Drot]
        - head_scaling 任意形状（如 [H] / [1,H,1]）都会被规整为一维 [H]
        - 线性插值在“时间维”上完成，并返回 [Tq/H/Tk 对应的 cos/sin] 为 [T, H, 1, Drot]
        """
        # 1) 确保 cache 至少有 2（为了 idx1=idx0+1 不越界）
        self._ensure_rope_cache(max(2, T_q, T_k), device, dtype)

        # cache: 默认 [Tbase, Drot]
        base_cos = self._cache_cos.to(device=device, dtype=dtype)
        base_sin = self._cache_sin.to(device=device, dtype=dtype)

        # 2) 位置（时间索引）
        pos_q = torch.arange(T_q, device=device, dtype=dtype)  # [Tq]
        pos_k = torch.arange(T_k, device=device, dtype=dtype)  # [Tk]

        # 3) 每个 head 的频率缩放，强制规整为一维 [H]
        s = torch.clamp(self.head_scaling, min=1e-3).reshape(-1).to(dtype)  # [H]
        H = int(s.numel())

        # ---- 内部：从 base_cos/base_sin 采样，返回 [T, H, 1, Drot] ----
        def sample_cos_sin(base_cos: torch.Tensor,
                        base_sin: torch.Tensor,
                        pos: torch.Tensor,   # [T]
                        s_1d: torch.Tensor   # [H]
                        ):
            # a) 统一 base_* 形状为 [Tbase, H, 1, Drot]
            if base_cos.dim() == 2:
                # [Tbase, Drot] → [Tbase, H, 1, Drot]
                Tbase, Drot = base_cos.shape
                base_cos = base_cos.unsqueeze(1).unsqueeze(2).expand(Tbase, H, 1, Drot)
                base_sin = base_sin.unsqueeze(1).unsqueeze(2).expand(Tbase, H, 1, Drot)
            elif base_cos.dim() == 3:
                # [Tbase, 1, Drot] → [Tbase, H, 1, Drot]
                Tbase, one, Drot = base_cos.shape
                if one != 1:
                    raise ValueError(f"Unexpected base_cos shape {tuple(base_cos.shape)}, expect second dim==1")
                base_cos = base_cos.unsqueeze(2).expand(Tbase, H, 1, Drot)
                base_sin = base_sin.unsqueeze(2).expand(Tbase, H, 1, Drot)
            elif base_cos.dim() == 4:
                Tbase, H0, _, Drot = base_cos.shape
                if H0 != H:
                    if H0 == 1:
                        base_cos = base_cos.expand(Tbase, H, 1, Drot)
                        base_sin = base_sin.expand(Tbase, H, 1, Drot)
                    else:
                        raise ValueError(f"head dim mismatch: base_cos.H={H0}, scaling.H={H}")
            else:
                raise ValueError(f"Unsupported base_cos ndim={base_cos.dim()}, shape={tuple(base_cos.shape)}")

            # 现在 base_*: [Tbase, H, 1, Drot]
            Tbase, _, _, Drot = base_cos.shape
            T = int(pos.shape[0])

            # b) 计算缩放位置 pos_scaled: [T, H]
            pos_scaled = pos.to(base_cos.dtype).unsqueeze(-1) * s_1d.unsqueeze(0)

            # 下/上界索引 [T, H]
            idx0 = torch.floor(pos_scaled).to(torch.long).clamp_(0, Tbase - 2)
            idx1 = idx0 + 1

            # 插值权重 [T, H, 1, 1]
            w = (pos_scaled - idx0.to(pos_scaled.dtype)).unsqueeze(-1).unsqueeze(-1)

            # c) 在时间维 gather，再还原回 [T, H, 1, Drot]
            cos_base = base_cos.permute(1, 0, 2, 3)  # [H, Tbase, 1, Drot]
            sin_base = base_sin.permute(1, 0, 2, 3)

            idx0_exp = idx0.permute(1, 0).unsqueeze(-1).unsqueeze(-1).expand(H, T, 1, Drot)
            idx1_exp = idx1.permute(1, 0).unsqueeze(-1).unsqueeze(-1).expand(H, T, 1, Drot)

            cos0 = torch.gather(cos_base, dim=1, index=idx0_exp).permute(1, 0, 2, 3)  # [T,H,1,Drot]
            cos1 = torch.gather(cos_base, dim=1, index=idx1_exp).permute(1, 0, 2, 3)
            sin0 = torch.gather(sin_base, dim=1, index=idx0_exp).permute(1, 0, 2, 3)
            sin1 = torch.gather(sin_base, dim=1, index=idx1_exp).permute(1, 0, 2, 3)

            # 线性插值（w: [T,H,1,1]；cos0/1: [T,H,1,Drot]）
            cos = cos0 * (1.0 - w) + cos1 * w   # [T, H, 1, Drot]
            sin = sin0 * (1.0 - w) + sin1 * w   # [T, H, 1, Drot]

            # === 关键：为了与 x:[B,T,H,Drot] 对齐，这里返回 [1, T, H, Drot]，便于按 batch 维广播 ===
            cos = cos.permute(2, 0, 1, 3).contiguous()  # [1, T, H, Drot]
            sin = sin.permute(2, 0, 1, 3).contiguous()  # [1, T, H, Drot]
            return cos, sin

        # 4) 分别为 Q / K 采样（返回 [Tq, H, 1, Drot] / [Tk, H, 1, Drot]）
        cos_q, sin_q = sample_cos_sin(base_cos, base_sin, pos_q, s)
        cos_k, sin_k = sample_cos_sin(base_cos, base_sin, pos_k, s)
        return cos_q, sin_q, cos_k, sin_k


    def forward(
        self,
        x_q: torch.Tensor,  # [B,Tq,D]
        x_kv: torch.Tensor,  # [B,Tk,D]
        attn_mask: Optional[torch.Tensor] = None,  # [Tq,Tk] (1/True=keep, 0/False=mask) or None
    ):
        B, Tq, _ = x_q.shape
        _, Tk, _ = x_kv.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x_q).view(B, Tq, H, Dh)
        k = self.k_proj(x_kv).view(B, Tk, H, Dh)
        v = self.v_proj(x_kv).view(B, Tk, H, Dh)

        # apply RoPE on the first rope_dim of q,k (rest unchanged)
        rope_dim = min(self.rope_dim, Dh)
        if rope_dim > 0:
            cos_q, sin_q, cos_k, sin_k = self._make_scaled_rope(
                Tq, Tk, x_q.device, x_q.dtype
            )  # [T,H,1,Drot]
            # split head dim into rope-part and plain-part
            q_rope, q_rest = q[..., :rope_dim], q[..., rope_dim:]
            k_rope, k_rest = k[..., :rope_dim], k[..., rope_dim:]
            q_rope = apply_rope(q_rope, cos_q, sin_q)  # [B,Tq,H,Drot]
            k_rope = apply_rope(k_rope, cos_k, sin_k)
            q = torch.cat([q_rope, q_rest], dim=-1)
            k = torch.cat([k_rope, k_rest], dim=-1)

        # scaled dot-product attention
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(Dh)  # [B,H,Tq,Tk]

        # causal mask (prevent seeing future in self-attn)
        if self.use_causal_mask and Tq == Tk and (attn_mask is None or attn_mask.shape != (Tq, Tk)):
            causal = torch.triu(torch.ones(Tq, Tk, device=x_q.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        # external mask: keep==1 or True, mask where 0/False
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                mask_bool = attn_mask == 0  # 0 -> mask
            else:
                mask_bool = ~attn_mask  # False -> mask
            attn_scores = attn_scores.masked_fill(mask_bool.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        if getattr(self, "save_attn", False):
            self.last_attn_nodrop = attn.detach()   # ① 先存一份“未dropout”的

        attn = self.attn_dropout(attn)
        if getattr(self, "save_attn", False):
            self.last_attn = attn.detach()          # ② 再存一份“已dropout”的

        y = torch.einsum("bhts,bshd->bthd", attn, v)  # [B,Tq,H,Dh]
        y = y.reshape(B, Tq, H * Dh)
        y = self.o_proj(y)
        y = self.proj_dropout(y)
        return y


# -----------------------------
# Transformer Blocks
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_dim: Optional[int] = None,
        head_freq_scaling: Optional[torch.Tensor] = None,
        learnable_head_scaling: bool = False,
        use_cross_attn: bool = False,
        use_causal_mask: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.self_attn = MultiHeadAttentionRoPE(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=ff_dropout,
            rope_theta=rope_theta,
            rope_dim=rope_dim,
            head_freq_scaling=head_freq_scaling,
            learnable_head_scaling=learnable_head_scaling,
            use_causal_mask=use_causal_mask,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=ff_dropout)
        self.ln2 = nn.LayerNorm(d_model)

        if use_cross_attn:
            self.cross_attn = MultiHeadAttentionRoPE(
                d_model=d_model,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                proj_dropout=ff_dropout,
                rope_theta=rope_theta,
                rope_dim=rope_dim,
                head_freq_scaling=head_freq_scaling,
                learnable_head_scaling=learnable_head_scaling,
                use_causal_mask=False,  # cross-attn 不用因果遮罩
            )
            self.ln_cross = nn.LayerNorm(d_model)

    def forward(self, x_enc: torch.Tensor, x_mem: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        # Self-Attn
        y = self.self_attn(self.ln1(x_enc), x_enc, attn_mask=attn_mask)
        x = x_enc + y
        # Cross-Attn (可选)
        if self.use_cross_attn and x_mem is not None:
            y2 = self.cross_attn(self.ln_cross(x), x_mem, attn_mask=None)
            x = x + y2
        # FFN
        z = self.ff(self.ln2(x))
        x = x + z
        return x


# -----------------------------
# RopeFormer Model (Encoder-only)
# -----------------------------
@dataclass
class RopeFormerConfig:
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 4
    d_ff: int = 512
    horizon: int = 1  # steps to predict
    rope_theta: float = 10000.0
    rope_dim: Optional[int] = None
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    use_cross_attn: bool = False
    use_causal_mask: bool = True
    pooling: Literal["last", "mean"] = "last"
    learnable_head_scaling: bool = False  # per-head RoPE scaling learnable
    # optional preset head scaling (list of length n_heads), else all ones
    head_scaling_init: Optional[list] = None
    # input projection
    d_in: int = 1  # feature dim per timestep
    # classification classes for direction, 1 => binary BCEWithLogits (one logit per step)
    dir_num_classes: int = 1

@register_method("ropeformer")
class RopeFormer(nn.Module):
    """
    Encoder-only RopeFormer with per-head RoPE scaling and two output heads:
      - Direction head: logits (binary default; set dir_num_classes=3 for up/flat/down)
      - Volatility head: log_sigma
    Forward returns: {"logits": [B,H] or [B,H,C], "log_sigma": [B,H]}
    支持用平铺kwargs初始化（与 main/registry 对齐），或用 cfg=RopeFormerConfig 初始化。
    """
    def __init__(self, cfg: Optional[RopeFormerConfig] = None, **kwargs):
        super().__init__()

        # 1) 允许平铺 kwargs（兼容 main.py 直接传 d_model/nhead/...）
        if cfg is None:
            # 将 enc_layers → num_layers；head_freq_scales → head_scaling_init
            num_layers = kwargs.pop("num_layers", kwargs.pop("enc_layers", 2))
            head_scaling_init = kwargs.pop("head_scaling_init", None)
            if head_scaling_init is None and "head_freq_scales" in kwargs:
                head_scaling_init = kwargs.pop("head_freq_scales")

            self.cfg = RopeFormerConfig(
                d_model=kwargs.pop("d_model", 256),
                n_heads=kwargs.pop("nhead", 8),
                num_layers=num_layers,
                d_ff=kwargs.pop("d_ff", kwargs.pop("ff_mult", 2)) if isinstance(kwargs.get("ff_mult", None), int) else kwargs.pop("d_ff", 512),
                horizon=kwargs.pop("horizon", 1),
                rope_theta=kwargs.pop("base_theta", kwargs.pop("rope_theta", 10000.0)),
                rope_dim=kwargs.pop("rope_dim", None),
                attn_dropout=kwargs.pop("attn_dropout", kwargs.pop("dropout", 0.0)),
                ff_dropout=kwargs.pop("ff_dropout", kwargs.pop("dropout", 0.0)),
                use_cross_attn=kwargs.pop("use_cross_attn", False),
                use_causal_mask=kwargs.pop("use_causal_mask", True),
                pooling=kwargs.pop("pooling", "last"),
                learnable_head_scaling=bool(kwargs.pop("learnable_head_scaling", False)),
                head_scaling_init=head_scaling_init,
                d_in=kwargs.pop("in_dim", kwargs.pop("d_in", 1)),
                dir_num_classes=int(kwargs.pop("dir_num_classes", 1)),
            )
            # 把未用的 kwargs 忽略掉（例如 task_mode 等），以最大兼容
        else:
            # 传入 dataclass 的原路径
            self.cfg = cfg

        # 2) 规范/校验 pooling
        if self.cfg.pooling not in ("last", "mean"):
            self.cfg.pooling = "last"

        # 3) 规范 head scaling 的长度与类型
        if self.cfg.head_scaling_init is not None:
            if isinstance(self.cfg.head_scaling_init, torch.Tensor):
                vals = self.cfg.head_scaling_init.detach().cpu().tolist()
            else:
                vals = list(self.cfg.head_scaling_init)
            if len(vals) != self.cfg.n_heads:
                raise ValueError(f"head_freq_scales/head_scaling_init 长度({len(vals)})必须等于 nhead({self.cfg.n_heads})")
            self._hs_init_list = vals
        else:
            self._hs_init_list = [1.0] * self.cfg.n_heads

        # ====== 原初始化逻辑（保持不变）======
        self.horizon = self.cfg.horizon
        self.in_proj = nn.Linear(self.cfg.d_in, self.cfg.d_model)

        hs_init_tensor = torch.tensor(self._hs_init_list, dtype=torch.float32)

        blocks = []
        for _ in range(self.cfg.num_layers):
            blocks.append(
                EncoderBlock(
                    d_model=self.cfg.d_model,
                    n_heads=self.cfg.n_heads,
                    d_ff=(self.cfg.d_ff if isinstance(self.cfg.d_ff, int) else int(self.cfg.d_model * float(self.cfg.d_ff))),
                    attn_dropout=self.cfg.attn_dropout,
                    ff_dropout=self.cfg.ff_dropout,
                    rope_theta=self.cfg.rope_theta,
                    rope_dim=self.cfg.rope_dim,
                    head_freq_scaling=hs_init_tensor.clone(),
                    learnable_head_scaling=self.cfg.learnable_head_scaling,
                    use_cross_attn=self.cfg.use_cross_attn,
                    use_causal_mask=self.cfg.use_causal_mask,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(self.cfg.d_model)

        # ==== [PATCH-A] 输出头：方向 logits + 波动 log_sigma ====
        # 读取必要配置（按你的命名调整）
        self.horizon = int(getattr(self, "horizon", kwargs.get("horizon", 1)))
        self.dir_num_classes = int(getattr(self, "dir_num_classes", kwargs.get("dir_num_classes", 1)))
        self.pooling = getattr(self, "pooling", kwargs.get("pooling", "last"))
        if self.pooling not in ("last", "mean"):
            self.pooling = "last"

        D = self.cfg.d_model if hasattr(self, "cfg") else self.d_model  # 兼容两种保存方式

        # 一个小的 MLP 头
        self.head_hidden = nn.Linear(D, D)
        self.head_act = nn.ReLU()

        # 方向头：二分类时输出 [B, H]；多分类时输出 [B, H, C]
        C = self.dir_num_classes
        self.dir_head = nn.Linear(D, self.horizon * (C if C > 0 else 1))

        # 波动头：输出每步的 log_sigma（对数标准差）
        self.vol_head = nn.Linear(D, self.horizon)

        # 新增：均值头 μ
        self.mu_head  = nn.Linear(D, self.horizon)
        self.mem_proj = None   # 懒创建：第一次看到 x_mem 时再根据其最后维度创建 Linear


    # ========= 放到 RopeFormer 类里，替换/新增这三个方法 =========

    def encode(
        self,
        x_enc: torch.Tensor,                      # [B,T_enc,D]
        x_mem: Optional[torch.Tensor] = None,     # [B,T_mem,D] or None
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        逐层编码；若配置打开 cross-attn 且传入 x_mem，则各层使用 cross-attn。
        """
        h = x_enc
        use_ca = bool(getattr(self.cfg, "use_cross_attn", False))
        m = x_mem if (x_mem is not None and use_ca) else None

        for blk in self.blocks:
            # EncoderBlock 需支持 x_mem=None 的分支（仅 self-attn）
            h = blk(h, x_mem=m, attn_mask=attn_mask)
        return h


    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """
        时间维聚合 -> [B, D]，后续 head 再映射到 [B,H] / [B,H,C]
        """
        if getattr(self.cfg, "pooling", "last") == "mean":
            z = h.mean(dim=1)        # [B,D]
        else:  # "last" or fallback
            z = h[:, -1, :]          # [B,D]

        # 小 MLP 激活（与 __init__ 里 head_hidden/head_act 对齐）
        z = self.head_act(self.head_hidden(z))   # [B,D]
        return z


    def forward(
        self,
        x_enc: torch.Tensor,               # [B, T_enc, d_in]
        x_dec: torch.Tensor,               # [B, T_dec, d_in]  (目前仅保持接口一致，可不参与)
        overlap_k: Optional[int] = None,   # 量化老路径的参数，这里保留但不使用
        x_mem: Optional[torch.Tensor] = None,  # [B, T_mem, d_exog]
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        统一走 μ/σ/方向 路径；不再调用 quantile_head。
        - 若 self.cfg.use_cross_attn=True 且给了 x_mem，则做 cross-attn；
        - 返回:
            vol+dir: {"mu":[B,H], "log_sigma":[B,H], "logits":[B,H] or [B,H,C]}
            dist:    {"mu":[B,H], "log_sigma":[B,H]}
        """
        B = x_enc.size(0)
        D = self.cfg.d_model

        # 1) 输入投影
        h_enc = self.in_proj(x_enc)    # [B, T_enc, D]
        _ = self.in_proj(x_dec)        # 保持接口一致（若未来用 context pooling 可用）

        # 2) 外生因子投影（懒创建，自动对齐列数）
        m = None
        if (x_mem is not None) and bool(getattr(self.cfg, "use_cross_attn", False)):
            in_dim_mem = x_mem.size(-1)
            if (self.mem_proj is None) or (self.mem_proj.in_features != in_dim_mem):
                self.mem_proj = nn.Linear(in_dim_mem, D).to(x_mem.device)
            m = self.mem_proj(x_mem)   # [B, T_mem, D]

        # 3) 编码 + 池化
        h = self.encode(h_enc, x_mem=m, attn_mask=attn_mask)  # [B, T_enc, D]
        z = self.pool(h)                                      # [B, D]

        # 4) 头部（μ / σ / 方向）
        H = int(self.horizon)
        mu        = self.mu_head(z).view(B, H)                              # [B,H]
        log_sigma = torch.clamp(self.vol_head(z).view(B, H), min=-20.0, max=5.0)  # [B,H]

        # dist 模式：只有 μ 和 σ
        if str(getattr(self, "task_mode", "vol+dir")) == "dist":
            return {"mu": mu, "log_sigma": log_sigma}

        # vol+dir：再加方向 logits
        C = int(getattr(self, "dir_num_classes", 1))
        if C <= 1:
            logits = self.dir_head(z).view(B, H)              # [B,H]  (sigmoid 外部做)
        else:
            logits = self.dir_head(z).view(B, H, C)           # [B,H,C] (softmax 外部做)
        return {"mu": mu, "log_sigma": log_sigma, "logits": logits}
