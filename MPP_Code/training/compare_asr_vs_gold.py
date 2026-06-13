"""
ASR vs gold experiments on MUStARD++.

1) ``whisper-experiment`` (default): Read **local** gold utterance text from ``mustard++_text.csv``
   (REFERENCE column ``SENTENCE``). Transcribe **dataset utterance clips** with Whisper —
   default search paths match the MUStARD++ Drive layout
   ``MPP_Code/data/final_utterance_videos/<KEY>.mp4`` (same as ``embedding_pipeline/media_paths.py``),
   plus ``MPP_Code/data/audio/``. **Whisper checkpoint weights** may download once (~tens–hundreds MB
   depending on model); that is separate from CSV data. Results: per-utterance WER vs gold,
   sklearn sarcasm baseline (TF-IDF + logistic, stratified K-fold weighted F1).
   If ``--asr-csv`` already has ASR rows, only utterance keys missing a non-empty ``asr_text`` are
   transcribed (resume); use ``--force-transcribe`` to redo everything. The CSV is saved after **each**
   clip so interruptions keep prior progress (~1200 clips on CPU may take hours: try ``tiny`` model and
   ``--max-samples``).

2) ``long-csv``: Compare two mustard++_text-style CSVs (utterance + context) with WER/CER
   per scene — no audio / no sklearn classification.

Requirements for ``whisper-experiment``::

    pip install openai-whisper torch pandas scikit-learn
    # System: ffmpeg on PATH (Whisper relies on it for many formats.)

Run from repository root (no subcommand → same as ``whisper-experiment``)::

    python MPP_Code/training/compare_asr_vs_gold.py

    python MPP_Code/training/compare_asr_vs_gold.py --audio-dir path/to/utterance_clips

    python MPP_Code/training/compare_asr_vs_gold.py whisper-experiment --audio-dir ...

    python MPP_Code/training/compare_asr_vs_gold.py long-csv --asr path/to/asr_same_format.csv

Flags before the subcommand (e.g. ``--gold-csv``) are treated as ``whisper-experiment``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding_pipeline.media_paths import default_utterance_video_dir


# ---------------------------------------------------------------------------
# Shared text / WER (long-csv mode + experiment)
# ---------------------------------------------------------------------------


def clean_text(x) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _levenshtein(a: Iterable[str], b: Iterable[str]) -> int:
    aa = list(a)
    bb = list(b)
    n, m = len(aa), len(bb)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if aa[i - 1] == bb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def word_error_rate(ref: str, hyp: str) -> float:
    """WER = edit distance(words) / |ref words|. NaN if reference has no words after clean."""
    r = clean_text(ref).split()
    h = clean_text(hyp).split()
    if len(r) == 0:
        return float("nan")
    return _levenshtein(r, h) / len(r)


def char_error_rate(reference: str, hypothesis: str) -> float | None:
    if len(reference) == 0:
        return None
    edits = _levenshtein(reference, hypothesis)
    return edits / max(len(reference), 1)


def _normalize_basic(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_remove_punct(s: str) -> str:
    s = _normalize_basic(s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: str, strip_punct: bool) -> str:
    return _normalize_remove_punct(s) if strip_punct else _normalize_basic(s)


def pairwise_metrics(
    gold_u: str, asr_u: str, gold_c: str, asr_c: str, strip_punct: bool
) -> Tuple[dict[str, float | None], dict[str, float | None]]:
    gu = normalize_text(gold_u, strip_punct)
    au = normalize_text(asr_u, strip_punct)
    gc = normalize_text(gold_c, strip_punct)
    ac = normalize_text(asr_c, strip_punct)
    utter = {
        "wer": word_error_rate(gu, au) if gu.split() else None,
        "cer": char_error_rate(gu, au),
    }
    context = {
        "wer": word_error_rate(gc, ac) if gc.split() else None,
        "cer": char_error_rate(gc, ac),
    }
    return utter, context


def summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name}: (no comparable rows)")
        return
    print(
        f"  {name}: mean={sum(values)/len(values):.4f} median={median(values):.4f} "
        f"n={len(values)} min={min(values):.4f} max={max(values):.4f}"
    )


# ---------------------------------------------------------------------------
# Whisper experiment: sklearn baseline
# ---------------------------------------------------------------------------


def eval_text_classifier(texts, labels, n_splits: int = 5, seed: int = 42, n_jobs: int = -1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, make_scorer
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
            ),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score, average="weighted")
    scores = cross_val_score(pipe, texts, labels, cv=cv, scoring=scorer, n_jobs=n_jobs)
    return scores.mean(), scores.std(), scores


_UTTERANCE_MEDIA_EXTS = ["wav", "mp3", "flac", "m4a", "aac", "mp4", "mkv", "webm", "mov"]


def find_utterance_media_file(media_dirs: Iterable[str], key: str) -> str | None:
    """Locate utterance clip for CSV KEY (e.g. ``1_10004_u``) under one of ``media_dirs``.

    Matches MUStARD++ Drive layout: ``final_utterance_videos/<KEY>.mp4`` (see ``media_paths.py``).
    """
    for audio_dir in media_dirs:
        ad = str(audio_dir)
        for ext in _UTTERANCE_MEDIA_EXTS:
            p = os.path.join(ad, f"{key}.{ext}")
            if os.path.exists(p):
                return p
        matches = []
        for ext in _UTTERANCE_MEDIA_EXTS:
            matches.extend(glob.glob(os.path.join(ad, f"**/{key}.{ext}"), recursive=True))
        if matches:
            return matches[0]
    return None


def default_utterance_media_dirs() -> list[Path]:
    """Dataset utterance clips (Google Drive extract), then optional flat ``audio/`` fallback."""
    return [
        default_utterance_video_dir(),
        REPO_ROOT / "MPP_Code" / "data" / "audio",
    ]


def load_existing_asr_map(asr_csv: Path) -> dict[str, str]:
    """``key -> raw asr_text`` from CSV; missing or bad file -> empty dict."""
    if not asr_csv.is_file():
        return {}
    try:
        t = pd.read_csv(asr_csv, dtype={"key": str})
    except Exception:
        return {}
    if "asr_text" not in t.columns or len(t) == 0:
        return {}
    out: dict[str, str] = {}
    for _, row in t.iterrows():
        k = str(row["key"]).strip()
        if not k:
            continue
        cell = row["asr_text"]
        out[k] = "" if pd.isna(cell) else str(cell)
    return out


def subsample_utterances_stratified(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Return up to ``n`` rows keeping approximate class balance via stratified shuffle split."""
    if n is None or n >= len(df):
        return df
    n = max(2, min(int(n), len(df)))
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx_arr, _ = next(splitter.split(np.zeros(len(df)), df["label"]))
    out = df.iloc[idx_arr].copy()
    return out.sort_values("key").reset_index(drop=True)


def cmd_whisper_experiment(ns: argparse.Namespace) -> None:
    gold_csv = Path(ns.gold_csv).expanduser().resolve()
    asr_csv = Path(ns.asr_csv).expanduser().resolve()
    out_dir = Path(ns.out_dir).expanduser().resolve()

    if ns.audio_dir is None:
        media_dirs = [p.resolve() for p in default_utterance_media_dirs()]
    else:
        media_dirs = [Path(ns.audio_dir).expanduser().resolve()]
    media_dir_strs = [str(d) for d in media_dirs]

    if not gold_csv.is_file():
        sys.exit(f"Gold CSV not found: {gold_csv}")

    print(f"Using local gold CSV only (nothing is downloaded): {gold_csv}")

    df = pd.read_csv(gold_csv, dtype={"KEY": str, "SCENE": str})
    if "Sarcasm" not in df.columns:
        sys.exit("Gold CSV must include a 'Sarcasm' column for the sarcasm baseline.")
    df = df[df["KEY"].astype(str).str.endswith("_u")].copy()
    df["key"] = df["KEY"].astype(str)
    df["gold_text"] = df["SENTENCE"].apply(clean_text)
    df["label"] = pd.to_numeric(df["Sarcasm"], errors="coerce").fillna(-1).astype(int)
    if (df["label"] < 0).any():
        bad = df.loc[df["label"] < 0, "KEY"].tolist()[:10]
        sys.exit(
            "Utterance rows must have numeric Sarcasm labels (0/1). "
            f"Problems (sample): {bad}"
        )

    if getattr(ns, "max_samples", None) is not None:
        df = subsample_utterances_stratified(df, ns.max_samples, getattr(ns, "subsample_seed", 42))
        print(
            f"Stratified subset: {len(df)} utterances (max_samples={ns.max_samples}, "
            f"subsample_seed={getattr(ns, 'subsample_seed', 42)})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    asr_csv.parent.mkdir(parents=True, exist_ok=True)

    keys = df["key"].tolist()
    existing = {} if ns.force_transcribe else load_existing_asr_map(asr_csv)

    trans_slots: list[str | None] = []
    for key in keys:
        if not ns.force_transcribe:
            raw = existing.get(key, "")
            if raw is not None and str(raw).strip():
                trans_slots.append(clean_text(str(raw)))
                continue
        trans_slots.append(None)

    pending_idx = [i for i, t in enumerate(trans_slots) if t is None]
    need_transcription = len(pending_idx) > 0
    cached_n = len(keys) - len(pending_idx)

    if cached_n > 0 and pending_idx:
        print(
            f"Resuming from {asr_csv}: reusing {cached_n} non-empty transcripts; "
            f"{len(pending_idx)} remaining for Whisper."
        )

    if need_transcription and len(keys) >= 200:
        print(
            "\nTIP: Full-corpus Whisper on CPU/GPU often takes a long time. Options:\n"
            "  --whisper-model tiny   (much faster than base/large)\n"
            "  --max-samples 100 --subsample-seed 42   (stratified subset for debugging)\n"
            "Ensure ffmpeg is installed and on PATH (first clip may stall briefly while ffmpeg loads).\n"
        )

    if need_transcription:
        print("Utterance media search paths (dataset layout: <KEY>.mp4 under final_utterance_videos):")
        for d in media_dirs:
            exists = "(missing - add Drive clips here)" if not d.is_dir() else ""
            print(f"  {d} {exists}")
        n_with_audio = sum(1 for k in keys if find_utterance_media_file(media_dir_strs, k) is not None)
        if n_with_audio == 0:
            sample_key = keys[0] if keys else "<KEY>"
            dirs_join = "\n".join(f"  - {d}" for d in media_dirs)
            sys.exit(
                "No utterance media matched search paths; will not download/load Whisper weights.\n\n"
                "Expected MUStARD++ utterance clips (from the project Drive folder), e.g.:\n"
                f"    .../MPP_Code/data/final_utterance_videos/{sample_key}.mp4\n\n"
                "Searched:\n"
                f"{dirs_join}\n\n"
                f"Formats tried: {', '.join(_UTTERANCE_MEDIA_EXTS)}\n\n"
                "Gold reference text is only read locally from mustard++_text.csv (not downloaded).\n"
                "Optional: copy WAV/MP3 into MPP_Code/data/audio as <KEY>.wav and re-run,\n"
                "or pass a custom folder:  --audio-dir PATH\n\n"
                "If you already have transcripts, provide  --asr-csv PATH  with columns key, asr_text\n"
                "(and omit --force-transcribe)."
            )

    if need_transcription:
        try:
            import whisper
        except ModuleNotFoundError as e:
            reason = "--force-transcribe requested" if ns.force_transcribe else ""
            if not reason:
                if not asr_csv.is_file():
                    reason = f"ASR CSV not found: {asr_csv}"
                elif len(pending_idx) == len(keys):
                    reason = f"need ASR transcripts for all {len(keys)} utterances"
                else:
                    reason = f"need Whisper for {len(pending_idx)} missing utterance transcripts ({asr_csv})"
            sys.exit(
                "The 'whisper' module was not found (install openai-whisper).\n\n"
                "  pip install openai-whisper\n"
                "  # or from repo root:  uv sync --extra asr\n\n"
                f"Whisper is only needed to transcribe audio ({reason}).\n"
                "If you already have an ASR transcript CSV (columns: key, asr_text), put it at\n"
                f"  {asr_csv}\n"
                "and run again without --force-transcribe - then Whisper is not required.\n\n"
                f"Original import error: {e}"
            )

        print(f"Transcribing with Whisper ({ns.whisper_model}) -> {asr_csv}")
        print(
            "Note: first run may download Whisper model weights (~140 MB). "
            "That download is the speech model, not mustard++_text.csv."
        )
        print("(Transcoding/decoding often needs ffmpeg on PATH.)")
        model = whisper.load_model(ns.whisper_model)

        def flush_asr_csv() -> None:
            pd.DataFrame({"key": keys, "asr_text": [(t if t is not None else "") for t in trans_slots]}).to_csv(
                asr_csv, index=False
            )

        found_audio = 0
        missing_audio = 0

        todo_n = len(pending_idx)
        for pos, j in enumerate(pending_idx, start=1):
            key = keys[j]
            audio_path = find_utterance_media_file(media_dir_strs, key)
            if audio_path is None:
                missing_audio += 1
                trans_slots[j] = ""
                print(f"[{pos}/{todo_n}] {key} -> (no media file)")
                flush_asr_csv()
                continue

            found_audio += 1
            print(f"[{pos}/{todo_n}] {key} -> transcribing...")
            sys.stdout.flush()
            try:
                result = model.transcribe(audio_path, language="en", fp16=False)
                text = clean_text(result["text"])
            except Exception as e:
                print(f"Transcription failed for {key}: {e}")
                text = ""

            trans_slots[j] = text
            flush_asr_csv()

        print(f"Saved ASR file: {asr_csv}")
        print(f"Utterance media transcribed (this pass): {found_audio} | missing on disk: {missing_audio}")

    else:
        print(
            f"All utterance keys already have non-empty transcripts in CSV "
            f"(use --force-transcribe to re-run Whisper): {asr_csv}"
        )

    asr = pd.read_csv(asr_csv, dtype={"key": str})
    if "asr_text" not in asr.columns:
        sys.exit("ASR CSV must contain columns: key, asr_text")

    asr["key"] = asr["key"].astype(str)
    asr["asr_text"] = asr["asr_text"].fillna("").apply(clean_text)

    merged = df.merge(asr[["key", "asr_text"]], on="key", how="inner").copy()
    merged = merged[(merged["gold_text"].str.len() > 0) & (merged["asr_text"].str.len() > 0)]

    if len(merged) == 0:
        n_asr_nonempty = int((asr["asr_text"].str.len() > 0).sum())
        sk = keys[0] if keys else "1_10004_u"
        sys.exit(
            "No usable rows: need both non-empty gold utterance text and non-empty ASR text.\n"
            f"ASR CSV rows with non-empty asr_text: {n_asr_nonempty} / {len(asr)}.\n\n"
            "Common causes:\n"
            "  - Utterance clips missing or wrong path (expect e.g. "
            f".../final_utterance_videos/{sk}.mp4 , or pass --audio-dir).\n"
            "  - Stale asr_transcripts.csv with all-empty asr_text; use "
            "--force-transcribe after adding media, or delete the CSV.\n\n"
            "The large download you may see on first Whisper run is only the model weights (~140 MB), "
            "not MUStARD++ CSV data."
        )

    merged["wer"] = [
        word_error_rate(g, a) for g, a in zip(merged["gold_text"], merged["asr_text"])
    ]

    gold_mean, gold_std, gold_scores = eval_text_classifier(
        merged["gold_text"], merged["label"], n_splits=ns.n_splits, n_jobs=ns.cv_n_jobs
    )
    asr_mean, asr_std, asr_scores = eval_text_classifier(
        merged["asr_text"], merged["label"], n_splits=ns.n_splits, n_jobs=ns.cv_n_jobs
    )

    print("\nDataset")
    print(f"Matched samples used: {len(merged)}")
    print(f"Sarcastic: {merged['label'].sum()} | Non-sarcastic: {(merged['label'] == 0).sum()}")

    print("\nASR quality (utterance)")
    print(f"Mean WER:   {merged['wer'].mean():.4f}")
    print(f"Median WER: {merged['wer'].median():.4f}")

    print("\nSarcasm detection (stratified K-fold weighted F1)")
    print(f"Gold transcripts : {gold_mean:.4f} ± {gold_std:.4f}")
    print(f"ASR transcripts  : {asr_mean:.4f} ± {asr_std:.4f}")
    print(f"Absolute drop    : {gold_mean - asr_mean:.4f}")

    print("\nFold-wise F1")
    print("Gold:", np.round(gold_scores, 4))
    print("ASR :", np.round(asr_scores, 4))

    summary = pd.DataFrame(
        [
            {
                "transcript_type": "gold",
                "weighted_f1_mean": gold_mean,
                "weighted_f1_std": gold_std,
                "mean_wer": np.nan,
            },
            {
                "transcript_type": "asr",
                "weighted_f1_mean": asr_mean,
                "weighted_f1_std": asr_std,
                "mean_wer": merged["wer"].mean(),
            },
        ]
    )

    results_path = out_dir / "asr_vs_gold_results.csv"
    rowwise_path = out_dir / "asr_vs_gold_rowwise.csv"
    summary.to_csv(results_path, index=False)
    merged[["key", "label", "gold_text", "asr_text", "wer"]].to_csv(rowwise_path, index=False)

    print("\nSaved files:")
    print(f"  {asr_csv}")
    print(f"  {results_path}")
    print(f"  {rowwise_path}")


def cmd_long_csv(ns: argparse.Namespace) -> None:
    from embedding_pipeline.load_text_csv import load_scene_texts

    gold_path = Path(ns.gold).resolve()
    asr_path = Path(ns.asr).resolve()
    if not gold_path.is_file():
        sys.exit(f"Gold CSV not found: {gold_path}")
    if not asr_path.is_file():
        sys.exit(f"ASR CSV not found: {asr_path}")

    gold: Dict[str, Tuple[str, str]] = load_scene_texts(str(gold_path))
    asr: Dict[str, Tuple[str, str]] = load_scene_texts(str(asr_path))

    gold_scenes = set(gold)
    asr_scenes = set(asr)
    common = gold_scenes & asr_scenes
    only_gold = sorted(gold_scenes - asr_scenes)
    only_asr = sorted(asr_scenes - gold_scenes)

    print(
        f"Gold scenes: {len(gold_scenes)} | ASR scenes: {len(asr_scenes)} | "
        f"Intersection: {len(common)}"
    )
    if only_gold:
        print(
            f"WARN: {len(only_gold)} scene(s) in gold but missing from ASR "
            f"(showing up to 10): {only_gold[:10]}"
        )
    if only_asr:
        print(
            f"WARN: {len(only_asr)} scene(s) in ASR but missing from gold "
            f"(showing up to 10): {only_asr[:10]}"
        )

    rows = []
    u_wers: list[float] = []
    u_cers: list[float] = []
    c_wers: list[float] = []
    c_cers: list[float] = []

    for scene in sorted(common):
        gu, gc = gold[scene]
        au, ac = asr[scene]
        utter, ctx = pairwise_metrics(gu, au, gc, ac, ns.strip_punct)
        uw, uc = utter["wer"], utter["cer"]
        cw, cc = ctx["wer"], ctx["cer"]
        if uw is not None:
            u_wers.append(float(uw))
        if uc is not None:
            u_cers.append(uc)
        if cw is not None:
            c_wers.append(float(cw))
        if cc is not None:
            c_cers.append(cc)

        rows.append(
            {
                "SCENE": scene,
                "utterance_wer": uw if uw is not None else float("nan"),
                "utterance_cer": uc if uc is not None else float("nan"),
                "context_wer": cw if cw is not None else float("nan"),
                "context_cer": cc if cc is not None else float("nan"),
            }
        )

    print("\nUtterance (target line):")
    summarize("WER", u_wers)
    summarize("CER", u_cers)
    print("\nContext (all context turns concatenated with newlines):")
    summarize("WER", c_wers)
    summarize("CER", c_cers)

    if ns.per_scene_csv is not None:
        out = Path(ns.per_scene_csv).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote per-scene table: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MUStARD++ ASR vs gold (Whisper + baseline, or CSV WER).")
    sub = p.add_subparsers(dest="command", required=True)

    wx = sub.add_parser(
        "whisper-experiment",
        help="Whisper transcripts + WER + TF-IDF/logistic sarcasm baseline.",
    )
    wx.add_argument(
        "--gold-csv",
        type=Path,
        default=REPO_ROOT / "mustard++_text.csv",
        help="Long-format MUStARD++ CSV with KEY / SENTENCE / Sarcasm (default: repo mustard++_text.csv).",
    )
    wx.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help=(
            "Single folder of utterance media named <KEY>.ext (KEY ends with _u). "
            "If omitted, search: (1) MPP_Code/data/final_utterance_videos (Drive layout: <KEY>.mp4), "
            "(2) MPP_Code/data/audio."
        ),
    )
    wx.add_argument(
        "--asr-csv",
        type=Path,
        default=REPO_ROOT / "MPP_Code" / "data" / "asr_transcripts.csv",
        help="Read/write Whisper output (columns: key, asr_text).",
    )
    wx.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "MPP_Code" / "stats" / "asr_vs_gold",
        help="Summary and row-wise WER+F1 artifacts.",
    )
    wx.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper checkpoint (tiny is much faster; base/small/large slower but usually more accurate).",
    )
    wx.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Re-run Whisper even if ASR CSV already exists.",
    )
    wx.add_argument("--n-splits", type=int, default=5, help="CV folds for the text baseline.")
    wx.add_argument(
        "--cv-n-jobs",
        type=int,
        default=-1,
        help="Joblib parallelism for CV (try 1 if you hit backend issues after torch).",
    )
    wx.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Stratified random subset size (deterministic seed; see --subsample-seed) for faster Whisper runs. "
            "Use a dedicated --asr-csv when experimenting so you do not truncate a full ASR CSV."
        ),
    )
    wx.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="Random seed for --max-samples stratified shuffle.",
    )
    wx.set_defaults(_run=cmd_whisper_experiment)

    lc = sub.add_parser(
        "long-csv",
        help="Compare gold vs ASR long CSVs with per-scene WER/CER (no Whisper).",
    )
    lc.add_argument(
        "--gold",
        type=Path,
        default=REPO_ROOT / "mustard++_text.csv",
        help="Gold full transcript CSV.",
    )
    lc.add_argument(
        "--asr",
        type=Path,
        required=True,
        help="Second CSV same format as gold (ASR-derived).",
    )
    lc.add_argument(
        "--strip-punct",
        action="store_true",
        help="Normalize (lowercase, optional strip punct) before WER/CER.",
    )
    lc.add_argument(
        "--per-scene-csv",
        type=Path,
        default=None,
        help="Optional path to save per-scene metrics.",
    )
    lc.set_defaults(_run=cmd_long_csv)

    return p


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv:
        argv = ["whisper-experiment"]
    elif argv[0] not in ("whisper-experiment", "long-csv") and argv[0].startswith("-"):
        # Allow: python compare_asr_vs_gold.py --gold-csv ...  (implies whisper-experiment)
        argv = ["whisper-experiment"] + argv
    args = parser.parse_args(argv)
    args._run(args)


if __name__ == "__main__":
    main()
