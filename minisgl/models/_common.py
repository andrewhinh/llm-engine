from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Type

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class BaseDecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        self_attn_ctor: Callable[..., BaseOP],
        mlp_ctor: Callable[[ModelConfig], BaseOP],
        self_attn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.self_attn = self_attn_ctor(
            config,
            layer_id,
            **(self_attn_kwargs or {}),
        )
        self.mlp = mlp_ctor(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class BaseTransformerModel(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        decoder_layer: Type[BaseDecoderLayer],
    ) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [decoder_layer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class CausalLMModel(BaseLLMModel):
    def __init__(self, config: ModelConfig, model_cls: Type[BaseOP]) -> None:
        self.model = model_cls(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["BaseDecoderLayer", "BaseTransformerModel", "CausalLMModel"]
