"""
Merge text, audio, and video pickles into one dict for MPP_Code training.

Each scene must appear in all three inputs. Output rows match training scripts:
  uText (1024), cText (1024), uAudio (291), cAudio (291), uVideo (2048), cVideo (2048)

Usage (repo root):
  uv run python -m embedding_pipeline.merge_multimodal_pickles
  uv run python -m embedding_pipeline.merge_multimodal_pickles \\
    --output MPP_Code/data/extracted_features/features_merged.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    base = REPO_ROOT / "MPP_Code" / "data" / "extracted_features"
    p.add_argument("--text", type=Path, default=base / "text_bart_large.pkl")
    p.add_argument("--audio", type=Path, default=base / "audio_291.pkl")
    p.add_argument("--video", type=Path, default=base / "video_resnet152.pkl")
    p.add_argument(
        "--output",
        type=Path,
        default=base / "features_merged.pkl",
    )
    args = p.parse_args()

    text_d = _load_pickle(args.text)
    audio_d = _load_pickle(args.audio)
    video_d = _load_pickle(args.video)

    keys_t, keys_a, keys_v = set(text_d), set(audio_d), set(video_d)
    scenes = sorted(keys_t & keys_a & keys_v, key=str)
    if not scenes:
        raise SystemExit("No common scene keys across the three pickles.")

    missing = []
    for name, s in ("text", keys_t), ("audio", keys_a), ("video", keys_v):
        extra = s - set(scenes)
        if extra:
            missing.append(f"{name}: {len(extra)} keys not in intersection")

    merged: dict = {}
    for scene in scenes:
        merged[scene] = {
            "uText": np.asarray(text_d[scene]["uText"], dtype=np.float32),
            "cText": np.asarray(text_d[scene]["cText"], dtype=np.float32),
            "uAudio": np.asarray(audio_d[scene]["uAudio"], dtype=np.float32),
            "cAudio": np.asarray(audio_d[scene]["cAudio"], dtype=np.float32),
            "uVideo": np.asarray(video_d[scene]["uVideo"], dtype=np.float32),
            "cVideo": np.asarray(video_d[scene]["cVideo"], dtype=np.float32),
        }

    s0 = merged[scenes[0]]
    assert s0["uText"].shape == (1024,) and s0["cText"].shape == (1024,)
    assert s0["uAudio"].shape == (291,) and s0["cAudio"].shape == (291,)
    assert s0["uVideo"].shape == (2048,) and s0["cVideo"].shape == (2048,)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Merged {len(merged)} scenes → {args.output}\n"
        f"  intersection: text∩audio∩video had {len(scenes)} keys."
    )
    if missing:
        for m in missing:
            print(f"  note: {m}")


if __name__ == "__main__":
    main()
