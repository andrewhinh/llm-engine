from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import BaseDecoderLayer, BaseTransformerModel, CausalLMModel

from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseDecoderLayer):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(
            config=config,
            layer_id=layer_id,
            self_attn_ctor=LlamaAttn,
            mlp_ctor=LlamaMLP,
        )


class LlamaModel(BaseTransformerModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, LlamaDecoderLayer)


class LlamaForCausalLM(CausalLMModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config, LlamaModel)


__all__ = ["LlamaForCausalLM"]
