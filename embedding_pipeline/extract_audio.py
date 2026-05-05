"""
Step 2: 291-D audio embeddings (utterance + context) per SCENE from mp4 files.

Usage (repo root):
  uv sync --extra embed
  uv run python -m embedding_pipeline.extract_audio
  (Uses ``ffmpeg`` on PATH if set; otherwise ``imageio-ffmpeg``'s bundled binary.)

Output: MPP_Code/data/extracted_features/audio_291.pkl
  dict[scene_id, {"uAudio": (291,), "cAudio": (291,)}]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from embedding_pipeline.audio_features import Audio291Extractor
from embedding_pipeline.load_text_csv import load_scene_texts
from embedding_pipeline.media_paths import context_video_path, utterance_video_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="291-D audio features for MUStARD++")
    p.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "mustard++_text.csv",
        help="CSV listing all SCENE ids",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "MPP_Code"
        / "data"
        / "extracted_features"
        / "audio_291.pkl",
        help="Output pickle path",
    )
    args = p.parse_args()

    scenes = sorted(load_scene_texts(str(args.csv)).keys())
    extractor = Audio291Extractor()

    data: dict = {}
    missing_u: list[str] = []
    missing_c: list[str] = []

    for scene in tqdm(scenes, desc="audio", unit="scene"):
        pu, pc = utterance_video_path(scene), context_video_path(scene)
        if not pu.is_file():
            missing_u.append(scene)
            continue
        if not pc.is_file():
            missing_c.append(scene)
            continue
        try:
            u_audio = extractor.from_media_path(str(pu))
            c_audio = extractor.from_media_path(str(pc))
        except Exception as e:
            raise RuntimeError(f"Failed scene={scene}: {e}") from e
        data[scene] = {"uAudio": u_audio, "cAudio": c_audio}

    if missing_u or missing_c:
        raise FileNotFoundError(
            f"Missing videos: utterance={len(missing_u)} context={len(missing_c)} "
            f"(e.g. {missing_u[:3]}{missing_c[:3]})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    s0 = next(iter(data.values()))
    print(
        f"Wrote {args.output} | scenes={len(data)} "
        f"uAudio/cAudio shape: {s0['uAudio'].shape}, {s0['cAudio'].shape}"
    )


if __name__ == "__main__":
    main()
