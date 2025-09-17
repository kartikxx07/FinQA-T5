# scripts/download_dataset.py

import os
from huggingface_hub import hf_hub_download

def download_finance_dataset(save_dir: str = "/Users/kartikayluthra/Desktop/finance-qa-/data"):
    repo_id = "Josephgflowers/Finance-Instruct-500k"
    filename = "train.json"

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    print("📥 Downloading train.json via HuggingFace hub...")

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=save_dir,
    )

    print(f"✅ Saved dataset to {out_path}")

if __name__ == "__main__":
    download_finance_dataset()


