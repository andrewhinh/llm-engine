import asyncio
import pathlib

from llmeng.modal import app, get_runtime_image, MINUTES

GPU_TYPE = "h200"
BLACKWELL_GPU_TYPE = "b200"

CI_IMAGE = (
    get_runtime_image(gpu_type=GPU_TYPE, extra_packages=("apache-tvm-ffi",))
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "tests"),
        "/root/tests",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "llmeng" / "kernel_new"),
        "/root/bench_kernel_new",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "llmeng" / "kernel_old"),
        "/root/bench_kernel_old",
    )
)
BLACKWELL_CI_IMAGE = (
    get_runtime_image(gpu_type=BLACKWELL_GPU_TYPE, extra_packages=("apache-tvm-ffi",))
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "tests"),
        "/root/tests",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "llmeng" / "kernel_new"),
        "/root/bench_kernel_new",
    )
    .add_local_dir(
        str(pathlib.Path(__file__).resolve().parent / "llmeng" / "kernel_old"),
        "/root/bench_kernel_old",
    )
)


@app.function(
    gpu=f"{GPU_TYPE}:2",
    image=CI_IMAGE,
)
async def pytest(target: str, pytest_args: str):
    import asyncio
    import shlex

    args = ["pytest", *shlex.split(pytest_args), target]
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd="/root",
    )
    return_code = await process.wait()
    if return_code != 0:
        raise RuntimeError(f"pytest exited with {return_code}")


@app.function(
    gpu=f"{BLACKWELL_GPU_TYPE}:2",
    image=BLACKWELL_CI_IMAGE,
)
async def pytest_blackwell(target: str, pytest_args: str):
    import asyncio
    import shlex

    args = ["pytest", *shlex.split(pytest_args), target]
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd="/root",
    )
    return_code = await process.wait()
    if return_code != 0:
        raise RuntimeError(f"pytest exited with {return_code}")


@app.local_entrypoint()
async def main(target: str = "tests", pytest_args: str = "-s -vv"):
    await asyncio.gather(
        pytest.remote.aio(target, pytest_args),
        pytest_blackwell.remote.aio(target, pytest_args),
    )
