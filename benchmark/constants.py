from typing import TypedDict


class ModelConfig(TypedDict):
    nnode: int
    gpu_type: str
    n_gpu: int
    server_kwargs: dict[str, int]


GUIDELLM_VERSION = "af89787"

MODEL_MATRIX: dict[str, ModelConfig] = {
    "Qwen/Qwen3-32B": {
        "nnode": 1,
        "gpu_type": "b200",
        "n_gpu": 1,
        "server_kwargs": {},
    },
    "Qwen/Qwen3-30B-A3B": {
        "nnode": 1,
        "gpu_type": "b200",
        "n_gpu": 1,
        "server_kwargs": {"cuda_graph_max_bs": -2},
    },
}

CONSTANT_RATES: dict[tuple[str, int], list[int | float]] = {
    ("b200", 1): [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
}

DATA_TYPES: dict[str, str] = {
    "128;1024": "prompt_tokens=128,output_tokens=1024",
    "256;2048": "prompt_tokens=256,output_tokens=2048",
    "512;512": "prompt_tokens=512,output_tokens=512",
    "512;4096": "prompt_tokens=512,output_tokens=4096",
    "1024;128": "prompt_tokens=1024,output_tokens=128",
    "1024;1024": "prompt_tokens=1024,output_tokens=1024",
    "2048;256": "prompt_tokens=2048,output_tokens=256",
    "2048;2048": "prompt_tokens=2048,output_tokens=2048",
    "4096;512": "prompt_tokens=4096,output_tokens=512",
}
