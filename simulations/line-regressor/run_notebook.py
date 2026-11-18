"""Execute the notebook in a fresh Python kernel, saving its figures and outputs."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
for variable, directory in {
    "MPLCONFIGDIR": "matplotlib",
    "IPYTHONDIR": "ipython",
    "JUPYTER_RUNTIME_DIR": "jupyter",
}.items():
    path = PROJECT / ".cache" / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(path))

import nbformat
from nbclient import NotebookClient


class ProgressClient(NotebookClient):
    """Also print notebook text outputs in the terminal while it is running."""

    def process_message(self, message, cell, cell_index):
        if message["msg_type"] == "stream":
            print(message["content"]["text"], end="", flush=True)
        return super().process_message(message, cell, cell_index)


def main():
    path = PROJECT / "npe_demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = ProgressClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(PROJECT)}},
    )
    client.execute(cwd=str(PROJECT))
    nbformat.write(notebook, path)
    print("\nSaved successfully executed notebook:", path)


if __name__ == "__main__":
    main()
