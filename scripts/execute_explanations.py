"""Execute the SHAP notebook with the current Python environment."""

import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def execute(output):
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/04_model_explainability.ipynb", as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(notebook, km=manager, timeout=300,
                   resources={"metadata": {"path": str(root)}}).execute(cleanup_kc=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed notebook: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="notebooks/04_model_explainability.ipynb")
    execute(parser.parse_args().output)
