"""RopeFormer: multi-scale rotary position encoding for financial time series.

Importing this package registers the ``ropeformer`` method in the registry.
"""

from .registry import register_method, build_method, list_methods
from .ropeformer import (
    RopeFormer,
    RopeFormerConfig,
    MultiHeadAttentionRoPE,
    EncoderBlock,
    FeedForward,
    build_rotary_cache,
    apply_rope,
    rotate_half,
)

__all__ = [
    "RopeFormer",
    "RopeFormerConfig",
    "MultiHeadAttentionRoPE",
    "EncoderBlock",
    "FeedForward",
    "build_rotary_cache",
    "apply_rope",
    "rotate_half",
    "register_method",
    "build_method",
    "list_methods",
]

__version__ = "0.1.0"
