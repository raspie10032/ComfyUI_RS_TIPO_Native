"""Optional prefetch; same downloader as first node execution."""
import argparse
import importlib.util
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('rs_tipo_download', Path(__file__).resolve().parents[1] / 'download.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_model(args.destination)
    print(f"TIPO native files ready: {args.destination}")
