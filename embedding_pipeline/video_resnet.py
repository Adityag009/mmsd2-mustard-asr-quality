"""
2048-D video embedding: Katna keyframe candidates → ImageNet ResNet-152 pool5,
mean-pooled over selected frames.

OpenCV frames are BGR; ResNet expects RGB + ImageNet normalization.
"""

from __future__ import annotations

import warnings
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet152_Weights, resnet152

try:
    from Katna.frame_extractor import FrameExtractor
    from Katna.image_selector import ImageSelector
except ImportError:  # some installs use lowercase package name
    from katna.frame_extractor import FrameExtractor  # type: ignore
    from katna.image_selector import ImageSelector  # type: ignore


def _uniform_sample_frames(path: str, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = min(num_frames, total)
    idxs = np.linspace(0, total - 1, n, dtype=int).tolist()
    out: List[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret and frame is not None:
            out.append(frame)
    cap.release()
    return out


def extract_keyframes_bgr(path: str, num_keyframes: int) -> List[np.ndarray]:
    """
    Katna candidate frames + diversity selection; uniform sampling if still short.
    """
    path = str(path)
    fe = FrameExtractor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = fe.extract_candidate_frames(path) or []

    selected: List[np.ndarray] = []
    if len(candidates) >= num_keyframes:
        selector = ImageSelector(n_processes=1)
        selected = list(selector.select_best_frames(candidates, num_keyframes))
        if len(selected) > num_keyframes:
            selected = selected[:num_keyframes]

    if len(selected) < num_keyframes:
        need = num_keyframes - len(selected)
        extra = _uniform_sample_frames(path, need + 8)
        for fr in extra:
            if len(selected) >= num_keyframes:
                break
            selected.append(fr)

    if not selected:
        selected = _uniform_sample_frames(path, num_keyframes)

    return selected[:num_keyframes]


class ResNet152VideoEmbedding(nn.Module):
    """ResNet-152 up to avgpool → (B, 2048)."""

    def __init__(self) -> None:
        super().__init__()
        m = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        pieces = [
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
            m.avgpool,
        ]
        self.features = nn.Sequential(*pieces)
        self.eval()

    @torch.inference_mode()
    def forward_stack(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return torch.flatten(z, 1)


class Video2048Extractor:
    def __init__(
        self,
        device: torch.device | None = None,
        batch_size: int = 8,
        num_keyframes: int = 12,
        use_fp16: bool = True,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.num_keyframes = num_keyframes
        self.use_fp16 = self.device.type == "cuda" and use_fp16
        weights = ResNet152_Weights.IMAGENET1K_V1
        self._preprocess = weights.transforms()
        self._model = ResNet152VideoEmbedding().to(self.device)
        self._model.eval()

    def _bgr_to_batch_tensor(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self._preprocess(pil))
        return torch.stack(tensors, dim=0)

    def from_video_path(self, path: str) -> np.ndarray:
        frames = extract_keyframes_bgr(path, self.num_keyframes)
        if not frames:
            return np.zeros(2048, dtype=np.float32)

        feats: list[torch.Tensor] = []
        for start in range(0, len(frames), self.batch_size):
            batch_bgr = frames[start : start + self.batch_size]
            x = self._bgr_to_batch_tensor(batch_bgr).to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z = self._model.forward_stack(x)
            else:
                z = self._model.forward_stack(x)
            feats.append(z.float().cpu())

        stacked = torch.cat(feats, dim=0)
        pooled = stacked.mean(dim=0).numpy().astype(np.float32)
        return pooled
