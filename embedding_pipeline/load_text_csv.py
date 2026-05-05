"""Build utterance + context strings from mustard++_text.csv (long format)."""

from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd

_CONTEXT_KEY = re.compile(r"_c_(\d+)$")


def _is_context_key(key: str) -> bool:
    return _CONTEXT_KEY.search(str(key)) is not None


def _context_sort_key(key: str) -> int:
    m = _CONTEXT_KEY.search(str(key))
    return int(m.group(1)) if m else -1


def load_scene_texts(csv_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns:
        scene_id -> (utterance_text, context_text)
        context_text: context turns joined with newlines, in order of _c_<n> suffix.
    """
    df = pd.read_csv(csv_path, dtype={"SCENE": str})
    if "KEY" not in df.columns or "SENTENCE" not in df.columns:
        raise ValueError("CSV must contain KEY and SENTENCE columns")

    out: Dict[str, Tuple[str, str]] = {}
    for scene, g in df.groupby("SCENE", sort=False):
        scene = str(scene)
        ctx_rows = g[g["KEY"].map(_is_context_key)].copy()
        utt_rows = g[~g["KEY"].map(_is_context_key)]

        if len(utt_rows) != 1:
            raise ValueError(
                f"SCENE {scene}: expected exactly one utterance row, got {len(utt_rows)}"
            )

        ctx_rows = ctx_rows.sort_values(
            "KEY", key=lambda s: s.map(_context_sort_key)
        )
        parts = []
        for sent in ctx_rows["SENTENCE"].astype(str):
            t = sent.strip()
            if t:
                parts.append(t)
        context_text = "\n".join(parts)

        utterance = str(utt_rows.iloc[0]["SENTENCE"]).strip()

        out[scene] = (utterance, context_text)

    return out
