import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset

dataset_dir = Path(__file__).resolve().parent / "dataset"
dataset_dir.mkdir(parents=True, exist_ok=True)

# Use HF mirror in environments with slow Hugging Face access.
ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(dataset_dir))
