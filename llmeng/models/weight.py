from __future__ import annotations

import glob
from typing import Dict

import safetensors
import torch
from tqdm import tqdm

from llmeng.distributed import get_tp_info
from llmeng.models.config import ModelConfig
from llmeng.utils import div_ceil, download_hf_weight, split_kv_heads


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor], model_config: ModelConfig
) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    _, kv_rank, kv_shard_count = split_kv_heads(model_config.num_kv_heads, n, r)
    SPLIT_DIM_0_LIST = [".gate_proj", ".up_proj"]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if key.count(".q_proj"):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif key.count(".k_proj") or key.count(".v_proj"):
            shard_state_dict[key] = value.chunk(kv_shard_count, dim=0)[kv_rank]
        elif any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_weight(
    model_path: str, model_config: ModelConfig, device: torch.device
) -> Dict[str, torch.Tensor]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}

    tp_info = get_tp_info()
    disable_tqdm = (tp_info.rank != 0) if tp_info.size > 1 else False
    device_str = str(device)

    for file in tqdm(sorted(files), desc="Loading weights", disable=disable_tqdm):
        with safetensors.safe_open(file, framework="pt", device=device_str) as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if tp_info.size > 1:
        state_dict = _shard_state_dict(state_dict, model_config)

    return _merge_state_dict(state_dict)
