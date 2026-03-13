from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import BaseDecoderLayer, BaseTransformerModel, CausalLMModel
from .utils import GatedMLP as Qwen2MLP
from .utils import RopeAttn as Qwen2Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen2DecoderLayer(BaseDecoderLayer):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(
            config=config,
            layer_id=layer_id,
            self_attn_ctor=Qwen2Attn,
            mlp_ctor=Qwen2MLP,
            self_attn_kwargs={"has_qk_norm": False, "has_attn_bias": True},
        )


class Qwen2Model(BaseTransformerModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, Qwen2DecoderLayer)


class Qwen2ForCausalLM(CausalLMModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, Qwen2Model)


__all__ = ["Qwen2ForCausalLM"]
