
import os
import json
import pandas as pd

def convert_json_to_csv(
    json_path: str = "/Users/kartikayluthra/Desktop/finance-qa-/data/train.json",
    save_dir: str = "/Users/kartikayluthra/Desktop/finance-qa-/data/processed"
):
    
    print("Loading JSONL file")

    df = pd.read_json(json_path, lines=True)

    print(f"Loaded {len(df)} records.")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "train.csv")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved CSV to {out_path}")

if __name__ == "__main__":
    convert_json_to_csv()
