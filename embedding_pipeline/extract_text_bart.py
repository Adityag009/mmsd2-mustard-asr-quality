"""
Step 1: extract 1024-D BART-Large embeddings (utterance + context) per SCENE.

Usage (from repo root, with uv):
  uv sync --extra embed
  uv run python -m embedding_pipeline.extract_text_bart --csv mustard++_text.csv
  # or: activate .venv then python -m embedding_pipeline.extract_text_bart ...

Output pickle: dict[scene_id, {"uText": np.ndarray(1024,), "cText": ...}]
Compatible with MPP_Code training (add video/audio keys in a later step).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel

from embedding_pipeline.load_text_csv import load_scene_texts

REPO_ROOT = Path(__file__).resolve().parents[1]


def _encoder(model):
    """BART may be wrapped (e.g. BartForConditionalGeneration); always use the encoder stack."""
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    if hasattr(model, "encoder"):
        return model.encoder
    raise TypeError("Model has no encoder; expected BART or compatible seq2seq.")


def _encode_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    use_fp16: bool,
) -> np.ndarray:
    """Mean pool over tokens after averaging the last 4 encoder hidden layers."""
    if not (text and text.strip()):
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    enc = _encoder(model)
    with torch.inference_mode():
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                enc_out = enc(**inputs, output_hidden_states=True)
        else:
            enc_out = enc(**inputs, output_hidden_states=True)

    # Tuple (embed + layer outs); use last 4 transformer layers
    hidden = enc_out.hidden_states
    stacked = torch.stack(hidden[-4:], dim=0).mean(dim=0)  # [1, seq, H]
    mask = inputs["attention_mask"].unsqueeze(-1).to(stacked.dtype)
    summed = (stacked * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    vec = (summed / denom).squeeze(0)

    return vec.float().cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description="BART-Large text embeddings for MUStARD++")
    p.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "mustard++_text.csv",
        help="Long-format CSV with SCENE, KEY, SENTENCE",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "MPP_Code"
        / "data"
        / "extracted_features"
        / "text_bart_large.pkl",
        help="Output pickle path",
    )
    p.add_argument(
        "--model",
        default="facebook/bart-large",
        help="Hugging Face model id",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:0, or cpu (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 on CUDA (use if you hit dtype issues)",
    )
    args = p.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(
            "CUDA requested but PyTorch has no CUDA; run: uv lock && uv sync --extra embed. "
            "Falling back to CPU.",
            flush=True,
        )
        device_str = "cpu"
    device = torch.device(device_str)
    use_fp16 = device.type == "cuda" and not args.no_fp16

    scene_texts = load_scene_texts(str(args.csv))
    print(f"Scenes: {len(scene_texts)} | device={device} fp16={use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # BartModel (not AutoModel / BartForConditionalGeneration): encoder hidden_states API is stable
    model = BartModel.from_pretrained(args.model)
    model.eval()
    model.to(device)

    data: dict = {}
    for scene, (utt, ctx) in tqdm(
        sorted(scene_texts.items()),
        desc="BART",
        unit="scene",
    ):
        data[scene] = {
            "uText": _encode_text(
                model, tokenizer, utt, device, args.max_length, use_fp16
            ),
            "cText": _encode_text(
                model, tokenizer, ctx, device, args.max_length, use_fp16
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample = next(iter(data.values()))
    u0, c0 = sample["uText"], sample["cText"]
    print(f"Wrote {args.output} | uText/cText shape: {u0.shape}, {c0.shape}")


if __name__ == "__main__":
    main()
