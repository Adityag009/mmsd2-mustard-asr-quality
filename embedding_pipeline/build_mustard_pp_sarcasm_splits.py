"""
Build files expected by MPP_Code/training/execute_sarcasm_mustard++.py from
mustard++_text.csv (repo root):

1. MPP_Code/data/final_datasets/mustard++_sarcasm_detection.csv
   - First column SCENE (training script uses index_col=0); includes SPEAKER (for speaker list).

2. MPP_Code/data/split_mustard_pp_sarcasm/train_{0-4}.csv, test_{0-4}.csv
   - Stratified 5-fold on binary Sarcasm label (seed=42)
   - Columns: SCENE, SAR, SPEAKER  (SAR matches ContentDataset in execute_sarcasm_mustard++.py)

These paths are gitignored on purpose; run this on any machine after cloning.

Usage (repo root):
  uv run python -m embedding_pipeline.build_mustard_pp_sarcasm_splits
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_context_key(key: str) -> bool:
    return bool(re.search(r"_c_\d+$", str(key)))


def main() -> None:
    src = REPO_ROOT / "mustard++_text.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Need {src}")

    df = pd.read_csv(src, dtype={"SCENE": str})
    utt = df[~df["KEY"].map(_is_context_key)].copy()
    if len(utt) != utt["SCENE"].nunique():
        raise ValueError("Expected exactly one utterance row per SCENE")

    if utt["Sarcasm"].isna().any():
        raise ValueError("Some utterances lack Sarcasm label")

    det = pd.DataFrame(
        {
            "SCENE": utt["SCENE"],
            "SPEAKER": utt["SPEAKER"],
            "Sarcasm": utt["Sarcasm"].astype(int),
            "Sarcasm_Type": utt["Sarcasm_Type"].astype(str),
        }
    )

    final_dir = REPO_ROOT / "MPP_Code" / "data" / "final_datasets"
    final_dir.mkdir(parents=True, exist_ok=True)
    det_path = final_dir / "mustard++_sarcasm_detection.csv"
    det.to_csv(det_path, index=False)
    print(f"Wrote {det_path} ({len(det)} rows)")

    split_dir = REPO_ROOT / "MPP_Code" / "data" / "split_mustard_pp_sarcasm"
    split_dir.mkdir(parents=True, exist_ok=True)

    fold_table = pd.DataFrame(
        {
            "SCENE": det["SCENE"],
            "SAR": det["Sarcasm"],
            "SPEAKER": det["SPEAKER"],
        }
    )
    X = fold_table["SCENE"].values
    y = fold_table["SAR"].values.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        fold_table.iloc[tr_idx].to_csv(split_dir / f"train_{fold}.csv", index=False)
        fold_table.iloc[te_idx].to_csv(split_dir / f"test_{fold}.csv", index=False)
    print(f"Wrote 5 folds under {split_dir}/")


if __name__ == "__main__":
    main()
