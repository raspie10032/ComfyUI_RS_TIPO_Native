"""Explicitly download only TIPO native model data; no remote Python is loaded."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "KBlueLeaf/TIPO-v2.1-1B-A200M"
REVISION = "f5a318524a4ab30cdbbf51816cf406170f454e65"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    snapshot_download(REPO, revision=REVISION, local_dir=str(args.destination),
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "README.md"])
    print(f"TIPO native files ready: {args.destination}")
