"""Default paths to MUStARD++ media under MPP_Code/data (flat layout)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_context_video_dir() -> Path:
    return REPO_ROOT / "MPP_Code" / "data" / "final_context_videos"


def default_utterance_video_dir() -> Path:
    return REPO_ROOT / "MPP_Code" / "data" / "final_utterance_videos"


def context_video_path(scene: str) -> Path:
    return default_context_video_dir() / f"{scene}_c.mp4"


def utterance_video_path(scene: str) -> Path:
    return default_utterance_video_dir() / f"{scene}_u.mp4"
