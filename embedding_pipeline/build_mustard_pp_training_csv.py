"""
Build files expected by MPP_Code/training from mustard++_text.csv (repo root):

1. MPP_Code/data/mustard_PP_utterance.csv
   - One row per SCENE; columns include SCENE, SPEAKER, Emo_E (1..9),
     Explicit_Emotion, Sarcasm, Sarcasm_Type

2. MPP_Code/data/splits_final_mustard++/train_{0-4}.csv, test_{0-4}.csv
   - Stratified 5-fold on Emo_E (seed=42)
   - Columns: SCENE, Emo_E, SPEAKER, SAR_T (SAR_T = Sarcasm_Type for result logs)

Usage (repo root):
  uv run python -m embedding_pipeline.build_mustard_pp_training_csv
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

    emotions = sorted(utt["Explicit_Emotion"].dropna().unique())
    if len(emotions) != 9:
        raise ValueError(f"Expected 9 explicit emotions, got {len(emotions)}: {emotions}")
    emo_to_id = {e: i + 1 for i, e in enumerate(emotions)}

    out = pd.DataFrame(
        {
            "SCENE": utt["SCENE"],
            "SPEAKER": utt["SPEAKER"],
            "Emo_E": utt["Explicit_Emotion"].map(emo_to_id),
            "Explicit_Emotion": utt["Explicit_Emotion"],
            "Sarcasm": utt["Sarcasm"],
            "Sarcasm_Type": utt["Sarcasm_Type"],
        }
    )
    if out["Emo_E"].isna().any():
        raise ValueError("Some utterances lack Explicit_Emotion")

    data_dir = REPO_ROOT / "MPP_Code" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    u_path = data_dir / "mustard_PP_utterance.csv"
    out.to_csv(u_path, index=False)
    print(f"Wrote {u_path} ({len(out)} rows)")
    print("Emo_E mapping (1-based, matches training script):")
    for k, v in sorted(emo_to_id.items(), key=lambda x: x[1]):
        print(f"  {v}: {k}")

    split_dir = data_dir / "splits_final_mustard++"
    split_dir.mkdir(parents=True, exist_ok=True)

    X = out["SCENE"].values
    y = out["Emo_E"].values.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_tables = out.assign(SAR_T=out["Sarcasm_Type"].astype(str))
    cols_split = ["SCENE", "Emo_E", "SPEAKER", "SAR_T"]

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        train = fold_tables.iloc[tr_idx][cols_split]
        test = fold_tables.iloc[te_idx][cols_split]
        train.to_csv(split_dir / f"train_{fold}.csv", index=False)
        test.to_csv(split_dir / f"test_{fold}.csv", index=False)
    print(f"Wrote 5 folds under {split_dir}/")


if __name__ == "__main__":
    main()
