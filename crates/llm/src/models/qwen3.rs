use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sigmoid;

use crate::models::layers::{
    Attention, KvCache, Linear, LmHead, RmsNorm, RotaryEmbedding, VocabEmbedding, get_context,
};
use crate::utils::tokenizer::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedShard {
    Q,
    K,
    V,
    Gate,
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedModuleMapping {
    pub source_fragment: &'static str,
    pub packed_module: &'static str,
    pub shard: PackedShard,
}

const PACKED_MODULES: [PackedModuleMapping; 5] = [
    PackedModuleMapping {
        source_fragment: "q_proj",
        packed_module: "qkv_proj",
        shard: PackedShard::Q,
    },
    PackedModuleMapping {
        source_fragment: "k_proj",
        packed_module: "qkv_proj",
        shard: PackedShard::K,
    },
    PackedModuleMapping {
        source_fragment: "v_proj",
        packed_module: "qkv_proj",
        shard: PackedShard::V,
    },
    PackedModuleMapping {
        source_fragment: "gate_proj",
        packed_module: "gate_up_proj",
        shard: PackedShard::Gate,
    },
    PackedModuleMapping {
        source_fragment: "up_proj",
        packed_module: "gate_up_proj",
        shard: PackedShard::Up,
    },
];

fn parse_layer_path(name: &str) -> Option<(usize, &str)> {
    let rest = name
        .strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("layers."))?;
    let (layer_idx, tail) = rest.split_once('.')?;
    let layer_idx = layer_idx.parse::<usize>().ok()?;
    Some((layer_idx, tail))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub hidden_act: String,
}

impl Qwen3Config {
    pub fn from_model_config(model_config: &ModelConfig) -> Result<Self> {
        ensure!(model_config.hidden_size > 0, "hidden_size must be positive");
        ensure!(
            model_config.num_attention_heads > 0,
            "num_attention_heads must be positive"
        );
        ensure!(
            model_config.num_hidden_layers > 0,
            "num_hidden_layers must be positive"
        );
        ensure!(model_config.vocab_size > 0, "vocab_size must be positive");

        let num_key_value_heads = model_config
            .num_key_value_heads
            .unwrap_or(model_config.num_attention_heads);
        ensure!(
            num_key_value_heads > 0,
            "num_key_value_heads must be positive"
        );
        ensure!(
            model_config
                .num_attention_heads
                .is_multiple_of(num_key_value_heads),
            "num_attention_heads must be divisible by num_key_value_heads"
        );

        let head_dim = model_config
            .head_dim
            .unwrap_or_else(|| model_config.hidden_size / model_config.num_attention_heads);
        ensure!(head_dim > 0, "head_dim must be positive");
        ensure!(
            model_config.hidden_size == model_config.num_attention_heads * head_dim,
            "hidden_size must equal num_attention_heads * head_dim"
        );

        let intermediate_size = model_config
            .intermediate_size
            .unwrap_or(model_config.hidden_size * 4);
        ensure!(intermediate_size > 0, "intermediate_size must be positive");
        ensure!(
            model_config.hidden_act == "silu",
            "only silu hidden_act is supported for qwen3 currently"
        );

        Ok(Self {
            hidden_size: model_config.hidden_size,
            num_attention_heads: model_config.num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            num_hidden_layers: model_config.num_hidden_layers,
            vocab_size: model_config.vocab_size,
            rms_norm_eps: model_config.rms_norm_eps,
            rope_theta: model_config.rope_theta,
            max_position_embeddings: model_config.max_position_embeddings,
            tie_word_embeddings: model_config.tie_word_embeddings,
            attention_bias: model_config.attention_bias,
            hidden_act: model_config.hidden_act.clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rotary: RotaryEmbedding,
    attn: Attention,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen3Attention {
    fn zeros(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        let q_out = config.num_attention_heads * config.head_dim;
        let kv_out = config.num_key_value_heads * config.head_dim;

        let q_proj = Linear::zeros(
            config.hidden_size,
            q_out,
            config.attention_bias,
            dtype,
            device,
        )?;
        let k_proj = Linear::zeros(
            config.hidden_size,
            kv_out,
            config.attention_bias,
            dtype,
            device,
        )?;
        let v_proj = Linear::zeros(
            config.hidden_size,
            kv_out,
            config.attention_bias,
            dtype,
            device,
        )?;
        let o_proj = Linear::zeros(config.hidden_size, config.hidden_size, false, dtype, device)?;

        let q_norm = if config.attention_bias {
            None
        } else {
            Some(RmsNorm::ones(
                config.head_dim,
                config.rms_norm_eps,
                dtype,
                device,
            )?)
        };
        let k_norm = if config.attention_bias {
            None
        } else {
            Some(RmsNorm::ones(
                config.head_dim,
                config.rms_norm_eps,
                dtype,
                device,
            )?)
        };

        let rotary = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;
        let attn = Attention::new(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            attn,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        })
    }

    fn load_weight(&mut self, name: &str, tensor: &Tensor) -> Result<bool> {
        match name {
            "q_proj.weight" => self.q_proj.set_weight(tensor.clone())?,
            "q_proj.bias" => self.q_proj.set_bias(tensor.clone())?,
            "k_proj.weight" => self.k_proj.set_weight(tensor.clone())?,
            "k_proj.bias" => self.k_proj.set_bias(tensor.clone())?,
            "v_proj.weight" => self.v_proj.set_weight(tensor.clone())?,
            "v_proj.bias" => self.v_proj.set_bias(tensor.clone())?,
            "o_proj.weight" => self.o_proj.set_weight(tensor.clone())?,
            "o_proj.bias" => self.o_proj.set_bias(tensor.clone())?,
            "q_norm.weight" => {
                if let Some(norm) = &mut self.q_norm {
                    norm.set_weight(tensor.clone())?;
                } else {
                    return Ok(false);
                }
            }
            "k_norm.weight" => {
                if let Some(norm) = &mut self.k_norm {
                    norm.set_weight(tensor.clone())?;
                } else {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn load_qkv_packed_shard(
        &mut self,
        packed_name: &str,
        shard: PackedShard,
        tensor: &Tensor,
    ) -> Result<bool> {
        match (packed_name, shard) {
            ("qkv_proj.weight", PackedShard::Q) => self.q_proj.set_weight(tensor.clone())?,
            ("qkv_proj.weight", PackedShard::K) => self.k_proj.set_weight(tensor.clone())?,
            ("qkv_proj.weight", PackedShard::V) => self.v_proj.set_weight(tensor.clone())?,
            ("qkv_proj.bias", PackedShard::Q) => self.q_proj.set_bias(tensor.clone())?,
            ("qkv_proj.bias", PackedShard::K) => self.k_proj.set_bias(tensor.clone())?,
            ("qkv_proj.bias", PackedShard::V) => self.v_proj.set_bias(tensor.clone())?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        slot_mapping: Option<&[i32]>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let tokens = hidden_states.dims2()?.0;
        let mut q =
            self.q_proj
                .forward(hidden_states)?
                .reshape((tokens, self.num_heads, self.head_dim))?;
        let mut k = self.k_proj.forward(hidden_states)?.reshape((
            tokens,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = self.v_proj.forward(hidden_states)?.reshape((
            tokens,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        if let Some(q_norm) = &self.q_norm {
            q = q_norm.forward(&q)?;
        }
        if let Some(k_norm) = &self.k_norm {
            k = k_norm.forward(&k)?;
        }

        let (q, k) = self.rotary.forward(positions, &q, &k)?;
        let context = get_context();
        let out = match kv_cache {
            Some(cache) => {
                if let Some(slot_mapping) = slot_mapping {
                    self.attn.write_kv_cache(&k, &v, slot_mapping, cache)?;
                }
                if context.is_prefill {
                    self.attn.forward_prefill(&q, &k, &v, true)?
                } else {
                    let block_tables_tensor = context
                        .block_tables
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("decode requires block_tables context"))?;
                    let block_tables =
                        block_tables_tensor.to_dtype(DType::U32)?.to_vec2::<u32>()?;
                    let context_lens_tensor = context
                        .context_lens
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("decode requires context_lens context"))?;
                    let context_lens_u32 =
                        context_lens_tensor.to_dtype(DType::U32)?.to_vec1::<u32>()?;
                    let context_lens = context_lens_u32
                        .into_iter()
                        .map(|len| len as usize)
                        .collect::<Vec<_>>();
                    self.attn.forward_decode(
                        &q,
                        cache,
                        &block_tables,
                        &context_lens,
                        context.block_size,
                    )?
                }
            }
            None => self.attn.forward_prefill(&q, &k, &v, true)?,
        };
        Ok(self
            .o_proj
            .forward(&out.reshape((tokens, self.num_heads * self.head_dim))?)?)
    }
}

#[derive(Debug, Clone)]
struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn zeros(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::zeros(
                config.hidden_size,
                config.intermediate_size,
                false,
                dtype,
                device,
            )?,
            up_proj: Linear::zeros(
                config.hidden_size,
                config.intermediate_size,
                false,
                dtype,
                device,
            )?,
            down_proj: Linear::zeros(
                config.intermediate_size,
                config.hidden_size,
                false,
                dtype,
                device,
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(hidden_states)?;
        let up = self.up_proj.forward(hidden_states)?;
        let gated = gate.broadcast_mul(&sigmoid(&gate)?)?;
        Ok(self.down_proj.forward(&gated.broadcast_mul(&up)?)?)
    }

    fn load_weight(&mut self, name: &str, tensor: &Tensor) -> Result<bool> {
        match name {
            "gate_proj.weight" => self.gate_proj.set_weight(tensor.clone())?,
            "gate_proj.bias" => self.gate_proj.set_bias(tensor.clone())?,
            "up_proj.weight" => self.up_proj.set_weight(tensor.clone())?,
            "up_proj.bias" => self.up_proj.set_bias(tensor.clone())?,
            "down_proj.weight" => self.down_proj.set_weight(tensor.clone())?,
            "down_proj.bias" => self.down_proj.set_bias(tensor.clone())?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn load_gate_up_packed_shard(
        &mut self,
        packed_name: &str,
        shard: PackedShard,
        tensor: &Tensor,
    ) -> Result<bool> {
        match (packed_name, shard) {
            ("gate_up_proj.weight", PackedShard::Gate) => {
                self.gate_proj.set_weight(tensor.clone())?
            }
            ("gate_up_proj.weight", PackedShard::Up) => self.up_proj.set_weight(tensor.clone())?,
            ("gate_up_proj.bias", PackedShard::Gate) => self.gate_proj.set_bias(tensor.clone())?,
            ("gate_up_proj.bias", PackedShard::Up) => self.up_proj.set_bias(tensor.clone())?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

#[derive(Debug, Clone)]
struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    fn zeros(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::zeros(config, dtype, device)?,
            mlp: Qwen3Mlp::zeros(config, dtype, device)?,
            input_layernorm: RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?,
            post_attention_layernorm: RmsNorm::ones(
                config.hidden_size,
                config.rms_norm_eps,
                dtype,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        slot_mapping: Option<&[i32]>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<(Tensor, Tensor)> {
        let (input_norm, input_residual) = if let Some(residual) = residual {
            self.input_layernorm
                .forward_with_residual(hidden_states, residual)?
        } else {
            let residual = hidden_states.clone();
            let normed = self.input_layernorm.forward(hidden_states)?;
            (normed, residual)
        };

        let attn_out = self
            .self_attn
            .forward(positions, &input_norm, slot_mapping, kv_cache)?;
        let (post_norm, post_residual) = self
            .post_attention_layernorm
            .forward_with_residual(&attn_out, &input_residual)?;
        let mlp_out = self.mlp.forward(&post_norm)?;
        Ok((mlp_out, post_residual))
    }

    fn load_weight(&mut self, name: &str, tensor: &Tensor) -> Result<bool> {
        if let Some(suffix) = name.strip_prefix("self_attn.") {
            return self.self_attn.load_weight(suffix, tensor);
        }
        if let Some(suffix) = name.strip_prefix("mlp.") {
            return self.mlp.load_weight(suffix, tensor);
        }
        match name {
            "input_layernorm.weight" => self.input_layernorm.set_weight(tensor.clone())?,
            "post_attention_layernorm.weight" => {
                self.post_attention_layernorm.set_weight(tensor.clone())?
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn load_packed_shard(
        &mut self,
        name: &str,
        shard: PackedShard,
        tensor: &Tensor,
    ) -> Result<bool> {
        if let Some(suffix) = name.strip_prefix("self_attn.") {
            return self.self_attn.load_qkv_packed_shard(suffix, shard, tensor);
        }
        if let Some(suffix) = name.strip_prefix("mlp.") {
            return self.mlp.load_gate_up_packed_shard(suffix, shard, tensor);
        }
        Ok(false)
    }
}

#[derive(Debug, Clone)]
struct Qwen3Model {
    embed_tokens: VocabEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
}

impl Qwen3Model {
    fn zeros(config: &Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::zeros(config, dtype, device)?);
        }
        Ok(Self {
            embed_tokens: VocabEmbedding::zeros(
                config.vocab_size,
                config.hidden_size,
                dtype,
                device,
            )?,
            layers,
            norm: RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?,
        })
    }

    fn forward_hidden(
        &self,
        positions: &Tensor,
        input_ids: &Tensor,
        slot_mapping: Option<&[i32]>,
        mut kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;
        for layer in &self.layers {
            let (next_hidden, next_residual) = layer.forward(
                positions,
                &hidden_states,
                residual.as_ref(),
                slot_mapping,
                kv_cache.as_deref_mut(),
            )?;
            hidden_states = next_hidden;
            residual = Some(next_residual);
        }

        if let Some(residual) = residual.as_ref() {
            let (normed, _) = self.norm.forward_with_residual(&hidden_states, residual)?;
            Ok(normed)
        } else {
            Ok(self.norm.forward(&hidden_states)?)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3ForCausalLM {
    config: Qwen3Config,
    model: Qwen3Model,
    lm_head: LmHead,
}

impl Qwen3ForCausalLM {
    pub fn zeros(config: Qwen3Config, dtype: DType, device: &Device) -> Result<Self> {
        let model = Qwen3Model::zeros(&config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            LmHead::new(model.embed_tokens.weight().clone())
        } else {
            LmHead::zeros(config.hidden_size, config.vocab_size, dtype, device)?
        };
        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    pub fn from_model_config(
        model_config: &ModelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let config = Qwen3Config::from_model_config(model_config)?;
        Self::zeros(config, dtype, device)
    }

    pub fn packed_modules_mapping() -> &'static [PackedModuleMapping] {
        &PACKED_MODULES
    }

    pub fn packed_module_for(weight_name: &str) -> Option<PackedModuleMapping> {
        Self::packed_modules_mapping()
            .iter()
            .copied()
            .find(|mapping| weight_name.contains(mapping.source_fragment))
    }

    pub fn load_weight(&mut self, name: &str, tensor: &Tensor) -> Result<bool> {
        match name {
            "model.embed_tokens.weight" | "embed_tokens.weight" => {
                self.model.embed_tokens.set_weight(tensor.clone())?;
                if self.config.tie_word_embeddings {
                    self.lm_head
                        .set_weight(self.model.embed_tokens.weight().clone())?;
                }
                return Ok(true);
            }
            "model.norm.weight" | "norm.weight" => {
                self.model.norm.set_weight(tensor.clone())?;
                return Ok(true);
            }
            "lm_head.weight" | "model.lm_head.weight" => {
                self.lm_head.set_weight(tensor.clone())?;
                return Ok(true);
            }
            _ => {}
        }

        if let Some((layer_idx, layer_tail)) = parse_layer_path(name) {
            ensure!(
                layer_idx < self.model.layers.len(),
                "layer index out of range in weight name: {}",
                name
            );
            return self.model.layers[layer_idx].load_weight(layer_tail, tensor);
        }
        Ok(false)
    }

    pub fn load_packed_shard(
        &mut self,
        packed_name: &str,
        shard: PackedShard,
        tensor: &Tensor,
    ) -> Result<bool> {
        if let Some((layer_idx, layer_tail)) = parse_layer_path(packed_name) {
            ensure!(
                layer_idx < self.model.layers.len(),
                "layer index out of range in packed weight name: {}",
                packed_name
            );
            return self.model.layers[layer_idx].load_packed_shard(layer_tail, shard, tensor);
        }
        Ok(false)
    }

    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        slot_mapping: Option<&[i32]>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let tokens = input_ids.dims1()?;
        ensure!(tokens > 0, "input_ids must not be empty");
        ensure!(
            positions.dims1()? == tokens,
            "positions must have one entry per token"
        );
        self.model
            .forward_hidden(positions, input_ids, slot_mapping, kv_cache)
    }

    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }

    pub fn forward_logits(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        slot_mapping: Option<&[i32]>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids, positions, slot_mapping, kv_cache)?;
        self.compute_logits(&hidden)
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        self.model.embed_tokens.weight().device()
    }
}
