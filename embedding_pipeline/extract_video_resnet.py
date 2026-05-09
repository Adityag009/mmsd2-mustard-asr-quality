"""
Step 3: 2048-D ResNet-152 video embeddings (utterance + context) per SCENE.

Uses Katna keyframe extraction + ImageNet ResNet-152 (pool5), mean over frames.

Usage (repo root):
  uv sync --extra embed
  uv run python -m embedding_pipeline.extract_video_resnet

Output: MPP_Code/data/extracted_features/video_resnet152.pkl
  dict[scene_id, {"uVideo": (2048,), "cVideo": (2048,)}]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from embedding_pipeline.load_text_csv import load_scene_texts
from embedding_pipeline.media_paths import context_video_path, utterance_video_path
from embedding_pipeline.video_resnet import Video2048Extractor

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="2048-D ResNet-152 video features")
    p.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "mustard++_text.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "MPP_Code"
        / "data"
        / "extracted_features"
        / "video_resnet152.pkl",
    )
    p.add_argument("--num-keyframes", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:0, or cpu (default: cuda if available)",
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 on CUDA",
    )
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(
            "CUDA was requested but this PyTorch build has no GPU support. "
            "From the repo root run: uv lock && uv sync --extra embed "
            "(Windows uses CUDA 13.0 wheels from the PyTorch index). "
            "Falling back to CPU.",
            flush=True,
        )
        device_str = "cpu"
    device = torch.device(device_str)
    scenes = sorted(load_scene_texts(str(args.csv)).keys())

    extractor = Video2048Extractor(
        device=device,
        batch_size=args.batch_size,
        num_keyframes=args.num_keyframes,
        use_fp16=not args.no_fp16,
    )

    data: dict = {}
    missing_u: list[str] = []
    missing_c: list[str] = []

    for scene in tqdm(scenes, desc="video", unit="scene"):
        pu, pc = utterance_video_path(scene), context_video_path(scene)
        if not pu.is_file():
            missing_u.append(scene)
            continue
        if not pc.is_file():
            missing_c.append(scene)
            continue
        try:
            u_vec = extractor.from_video_path(str(pu))
            c_vec = extractor.from_video_path(str(pc))
        except Exception as e:
            raise RuntimeError(f"Failed scene={scene}: {e}") from e
        data[scene] = {"uVideo": u_vec, "cVideo": c_vec}

    if missing_u or missing_c:
        raise FileNotFoundError(
            f"Missing videos: utterance={len(missing_u)} context={len(missing_c)} "
            f"(e.g. u:{missing_u[:3]} c:{missing_c[:3]})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    s0 = next(iter(data.values()))
    print(
        f"Wrote {args.output} | scenes={len(data)} "
        f"uVideo/cVideo shape: {s0['uVideo'].shape}, {s0['cVideo'].shape}"
    )


if __name__ == "__main__":
    main()
