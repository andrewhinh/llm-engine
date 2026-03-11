import modal

MODEL_DIR = "/models"
MODEL_VOLUME_NAME = "mini-sglang-models"
MINUTES = 60  # seconds

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

app = modal.App("mini-sglang", volumes={MODEL_DIR: model_volume})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install(
        "accelerate",
        "apache-tvm-ffi>=0.1.4",
        "fastapi",
        "flashinfer-python>=0.5.3",
        "huggingface_hub[hf_transfer]",
        "modelscope",
        "msgpack",
        "openai",
        "prompt_toolkit",
        "pyarrow",
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "pyzmq",
        "quack-kernels",
        "sgl_kernel>=0.3.17.post1",
        "torch<2.10.0",
        "transformers>=4.56.0,<=4.57.3",
        "uvicorn",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("minisgl")
    .add_local_dir("minisgl/kernel/csrc", remote_path="/root/minisgl/kernel/csrc")
)


def resolve_model_path(
    model_path: str,
) -> str:
    import os

    if os.path.exists(model_path):
        return model_path

    local_name = model_path.replace("/", "--")
    local_path = os.path.join(MODEL_DIR, local_name)

    if not os.path.exists(local_path):
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(model_path, local_dir=local_path)
        model_volume.commit()

    return local_path
