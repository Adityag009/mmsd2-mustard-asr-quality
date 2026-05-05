# Multimodal sarcasm detection — MMSD × adapted MUStARD++

This repository is the **working codebase** for our project on multimodal sarcasm detection, including **automatic speech recognition (ASR)** experiments and benchmarking with **MMSD2.0-style** setups on an **adapted MUStARD++** split (subset / modality reductions described below). The corpus and prior feature pipelines come from **[MUStARD++](https://aclanthology.org/2022.lrec-1.137/)** (LREC 2022: *A Multimodal Corpus for Emotion Recognition in Sarcasm*).

**Suggested names for your new GitHub repo** (pick one that fits your course/lab conventions):

| Style | Examples |
|--------|----------|
| Model + corpus | `mmsd2-mustardpp`, `MMSD2.0-adapted-mustard-plus-plus` |
| Task-first | `multimodal-sarcasm-asr`, `sarcasm-mmsd-asr-benchmark` |
| Short lab label | `mmsd-lab-mustardpp`, `mustard-image-text-asr` |

Use this README to orient the machine, install dependencies, and run training or extraction. **Training flags and paths** remain in [`MPP_Code/README.md`](MPP_Code/README.md).

---

## Research questions (working set)

Directions we track in experiments (aligned with our **MMSD2.0** architecture on the **adapted MUStARD++** benchmark):

- **Modality ablation (image–text):** Under MMSD2.0, which view contributes most to sarcasm accuracy—**text only**, **image only**, or **both** with interaction/fusion?
- **Modality weighting:** How does **varying the weight** of text vs image/video vs audio channels change detection performance?
- **Data scale:** How does **training set size** affect MMSD2.0 on the adapted benchmark?
- **ASR vs gold text:** How does **automatic speech recognition** transcript quality compare to the **original MUStARD++ transcripts** for sarcasm detection on the adapted dataset?

---

## From rich MUStARD++ to “one image + text”: what you lose

Full MUStARD++ instances are **dialogue-grounded**: an **utterance** plus **conversational context**, with **video** (and often **audio**) over time—not a single static frame. If you reduce that to **one still image + one text string**, you typically give up:

| Signal | Why it matters for sarcasm |
|--------|-----------------------------|
| **Temporal video** | Prosody-like facial motion, timing, reaction shots, and **irony in delivery** are spread over frames; one keyframe can miss the **punchline moment** or the contrast between neutral setup and exaggerated face. |
| **Audio / speech prosody** | Stress, tempo, laughter, sarcastic tone—“say it friendly, mean it mean”—is largely lost unless encoded elsewhere. This ties directly to your **ASR** track: transcripts drop **how** something was said, not only wording errors. |
| **Context turns** | Sarcasm often depends on **prior dialogue**; utterance-only or de-contextualized text removes **shared knowledge** and **setup** cues the model saw in richer settings. |
| **Scene semantics over time** | A single crop may omit **scene type**, **relations between speakers**, or **visual punchlines** that unfold across shots. |

**Likely effects on detection:** Models may lean more on **lexical polarity** (“great”, “love it”) paired with **static visual sentiment** (“traffic jam”), which helps some ironic pairs but hurts cases where sarcasm lives in **timing, prosody, or multi-turn coherence**. Absolute scores are **not directly comparable** to full video+context+V/A baselines—you are measuring performance on an **adapted, information-poor** view of the same corpus.

---

## What is in this repository?

| Area | Contents |
|------|----------|
| [`MPP_Code/`](MPP_Code/) | PyTorch models, training scripts, CSV splits, extracted feature pickles consumed by training |
| [`embedding_pipeline/`](embedding_pipeline/) | Scripts to build CSVs / extract multimodal embeddings (see “Feature extraction”) |
| `mustard++_text.csv` | Text + annotations export (referenced by embedding scripts via repo root paths) |

The multimodal corpus pairs each **utterance** with **context** from TV dialogue. Instances are annotated with sarcasm, emotion-related labels, valence/arousal, etc.; see the table below.

| Column | Description |
|--------|--------------|
| Sarcasm | 0 / 1 |
| Sarcasm_Type | Type if sarcastic, else `None` |
| Implicit_Emotion | Hidden / implied emotion |
| Explicit_Emotion | Surface emotion |
| Valence | 1–9 |
| Arousal | 1–9 |

**Videos** are not stored in Git (large assets). Obtain them via the **[Google Drive folder](https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW?usp=sharing)** linked in the original release.

---

## Quick start for group members

### 1. Clone and enter the repo

```powershell
git clone <your-new-repo-url>
cd <your-repo-folder>
```

### 2. Python environment

Recommended: **[uv](https://docs.astral.sh/uv/)** (reads `pyproject.toml`).

```powershell
uv sync
```

Alternatively: install **Python 3.10+**, then PyTorch compatible with your GPU/CPU ([pytorch.org](https://pytorch.org/get-started/locally/)), plus `numpy`, `pandas`, `scikit-learn` (same versions as `pyproject.toml` if you care about reproducibility).

Optional embeddings stack (BERT, Librosa, OpenSMILE, video deps):

```powershell
uv sync --extra embed
```

### 3. Run training — working directory + `PYTHONPATH`

All training scripts expect the **repository root** as current working directory, and **`MPP_Code` on `PYTHONPATH`** (they use `from models import ...`).

**Windows (PowerShell), from repo root:**

```powershell
$env:PYTHONPATH = "$PWD\MPP_Code"
python MPP_Code\training\execute_classification_explicit.py -s n -m VTA -c y
```

**Linux / macOS:**

```bash
export PYTHONPATH="$(pwd)/MPP_Code"
python MPP_Code/training/execute_classification_explicit.py -s n -m VTA -c y
```

More tasks (implicit emotion, regression, sarcasm) and exact data paths: [`MPP_Code/README.md`](MPP_Code/README.md). Use `-h` on any script for flags.

Training defaults to CUDA (`cuda:0`, etc.). For CPU-only, change `torch.device(...)` inside the relevant script(s).

---

## Feature extraction (`embedding_pipeline/`)

Embedding scripts default to **`mustard++_text.csv`** at repo root (`REPO_ROOT`). Typical flow:

1. Normalize media paths via `embedding_pipeline/media_paths.py` (`--help` for options).
2. Run extractors, e.g. `extract_text_bart`, `extract_audio`, `extract_video_resnet` (see module docstrings / `uv run python -m embedding_pipeline.<module> --help`).
3. Merge modalities if needed (e.g. `merge_multimodal_pickles.py`).

Install the `embed` optional dependency group before running these.

---

## Experiment outputs

Training writes under `MPP_Code/log/`, `MPP_Code/charts/`, `MPP_Code/stats/`, and `MPP_Code/predictions/`; these are **not** listed in [`.gitignore`](.gitignore), so you can **commit** logs, stats, and predictions with the rest of the repo if you want collaborators to see exact runs.

Pre-extracted tensors may live under `MPP_Code/data/extracted_features/` (`.pkl` files). They can be **large**; if the remote grows too heavy, switch to Git LFS or a shared Drive folder and document a download step here.

---

## Suggested collaboration workflow

- **Branch per experiment** or per person; merge when results are stable.
- **Document** in the PR or a short `experiments/<name>.md` (only if the team wants a log—keep it minimal) the script, flags, seed, and commit hash.

## What we commit vs keep local

Remote includes **experiment outputs** (`MPP_Code/log/`, `MPP_Code/charts/`, `MPP_Code/stats/`, `MPP_Code/predictions/` when you add them), **`MPP_Code/data/extracted_features/*.pkl`**, and the **training code**.

We **omit** **`uv.lock`** (install from `pyproject.toml` locally), **`MPP_Code/data` CSV splits / labels**, **local video clips**, and anything under **`augmented_utterance`** (see [.gitignore](.gitignore)). Teammates should obtain originals from the MUStARD++ release / Drive / your lab share and regenerate or copy paths locally.

---

## Project layout (summary)

```
<repo-root>/
  README.md                 ← you are here
  pyproject.toml            ← core deps (no uv.lock on remote; run uv sync locally)
  mustard++_text.csv
  embedding_pipeline/       ← feature extraction helpers
  MPP_Code/
    README.md               ← training: paths, scripts, checklist
    models/
    training/
    data/
      extracted_features/   ← pickles used by training (when present)
      ...
```

---

## Attribution

This corpus and codebase follow the MUStARD++ LREC 2022 paper. Use the citation from ACL Anthology in academic work citing the dataset ([link above](https://aclanthology.org/2022.lrec-1.137/)).
