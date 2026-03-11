from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import BaseDecoderLayer, BaseTransformerModel, CausalLMModel
from .utils import MoEMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseDecoderLayer):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(
            config=config,
            layer_id=layer_id,
            self_attn_ctor=Qwen3Attn,
            mlp_ctor=Qwen3MLP,
            self_attn_kwargs={"has_qk_norm": True},
        )


class Qwen3Model(BaseTransformerModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, Qwen3DecoderLayer)


class Qwen3MoeForCausalLM(CausalLMModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, Qwen3Model)


__all__ = ["Qwen3MoeForCausalLM"]
