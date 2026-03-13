import pathlib

from llmeng.modal import app, image

CI_IMAGE = image.add_local_dir(
    str(pathlib.Path(__file__).resolve().parent / "tests"),
    "/root/tests",
)


@app.function(
    gpu="any:2",
    image=CI_IMAGE,
)
def pytest():
    import subprocess

    repo_root = pathlib.Path("/root")
    subprocess.run(["pytest", "-q", str(repo_root / "tests")], check=True)


@app.local_entrypoint()
def main():
    pytest.remote()
