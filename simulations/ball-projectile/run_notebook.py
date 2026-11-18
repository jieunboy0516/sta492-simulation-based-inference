"""Run the tutorial from a fresh kernel and save its outputs."""

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
for variable, folder in {
    "MPLCONFIGDIR": "matplotlib", "IPYTHONDIR": "ipython", "JUPYTER_RUNTIME_DIR": "jupyter"
}.items():
    path = PROJECT / ".cache" / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(path))

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class ProgressClient(NotebookClient):
    def process_message(self, message, cell, cell_index):
        if message["msg_type"] == "stream":
            print(message["content"]["text"], end="", flush=True)
        return super().process_message(message, cell, cell_index)


def main():
    path = PROJECT / "projectile_demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Always use the interpreter running this script, including a local .venv.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = ProgressClient(
        notebook, km=manager, timeout=1800, allow_errors=False,
        resources={"metadata": {"path": str(PROJECT)}},
    )
    client.execute(cwd=str(PROJECT))
    nbformat.write(notebook, path)
    print("Saved successfully executed notebook: projectile_demo.ipynb")


if __name__ == "__main__":
    main()
