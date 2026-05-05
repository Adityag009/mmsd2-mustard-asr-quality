# MUStARD++ — How to run the training code locally

This folder contains PyTorch models (`models/`) and training scripts (`training/`) for multimodal sarcasm detection and emotion tasks on MUStARD++. The main dataset description and video link are in the [repository root README](../README.md).

---

## 1. Requirements

- **Python 3.8+** (3.10+ recommended)
- **PyTorch** + **torchvision** (install a build that matches your system from [pytorch.org](https://pytorch.org/get-started/locally/))
- **numpy**, **pandas**, **scikit-learn**

```bash
pip install torch torchvision numpy pandas scikit-learn
```

With **[uv](https://docs.astral.sh/uv/)** from the **repository root** (where `pyproject.toml` lives):

```bash
uv sync
```

That creates or updates `.venv` and installs from `pyproject.toml`. Optionally run **`uv lock`** locally if you want a personal `uv.lock`; this remote does **not** track `uv.lock` (see root `README`).

The scripts assume a **CUDA GPU** by default (`cuda:0`, `cuda:1`, etc.). To train on CPU you would need to edit each script and change `torch.device("cuda:...")` to `torch.device("cpu")`.

---

## 2. Working directory and Python path

All commands below assume your **current working directory is the repository root** (the folder that contains `MPP_Code/`), because paths like `MPP_Code/data/...` are written relative to that root.

Training scripts use `from models import ...`, so **`MPP_Code` must be on `PYTHONPATH`**.

**Windows (PowerShell), from the repo root:**

```powershell
cd path\to\your-repo-clone
$env:PYTHONPATH = "$PWD\MPP_Code"
```

**Linux / macOS:**

```bash
cd /path/to/your-repo-clone
export PYTHONPATH="$(pwd)/MPP_Code"
```

---

## 3. Data you need

Training reads **CSV metadata** plus **pickled feature tensors**:

- **`MPP_Code/data/extracted_features/*.pkl`** — versioned here; you still must add **CSVs/splits locally** wherever the script expects them (unless you rewrote paths).
- **`MPP_Code/data/*.csv`** and **`.../splits_*/*.csv`** (and local **video clips** under `MPP_Code/data/…`) — **not** in this remote ([`.gitignore`](../.gitignore)). Obtain from the MUStARD++ **[Drive folder](https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW?usp=sharing)**, replicate an older experiment machine, or copy from a teammate; exact filenames vary by script (see section 5). Some pickles may live under subfolders such as **`an_merged/`** depending on which training file you run.

Regenerate embeddings with `embedding_pipeline/` at the repo root (root [`README.md`](../README.md)).

If a path is missing, the script will fail at `read_csv` or `open(...pickle)`.

---

## 4. Log and output folders

Scripts write logs, charts, stats, and predictions under paths such as:

- `MPP_Code/log/...`
- `MPP_Code/charts/...`
- `MPP_Code/stats/...`
- `MPP_Code/predictions/...`

Create the subfolders your script uses if you get “No such file or directory” when opening a log file.

---

## 5. Training scripts and required arguments

Run with `python` from the **repository root** (with `PYTHONPATH` set as above).

Common **required** flags (all scripts):

| Flag | Meaning |
|------|--------|
| `-s` / `--speaker` | `y` = speaker-dependent, `n` = speaker-independent |
| `-m` / `--mode` | Modality code: e.g. `VTA` (video+text+audio), or `VT`, `VA`, `TA`, `V`, `T`, `A` |
| `-c` / `--context` | `y` = use context, `n` = utterance only |

Other useful flags (names vary slightly between scripts): learning rate, batch size, patience, dropout, GPU id (`-gpu`), etc. Use `-h` on any script for the full list.

### Explicit emotion classification

```bash
python MPP_Code/training/execute_classification_explicit.py -s n -m VTA -c y
```

Expects e.g. `MPP_Code/data/mustard_PP_utterance.csv`, feature pickles, and splits under `MPP_Code/data/splits_final_mustard++/`.

### Implicit emotion classification

```bash
python MPP_Code/training/execute_classification_implicit.py -s n -m VTA -c y
```

Uses paths such as `MPP_Code/data/final_datasets/augmented_sarcastic_utterances.csv` and merged feature pickles (see the `temp = open(...)` line near the top of the file for the active pickle).

### Valence / arousal regression

```bash
python MPP_Code/training/execute_regression.py -s n -m VTA -c y
```

Expects `MPP_Code/data/mustard_PP_utterance.csv` and the pickle named in that script.

### Sarcasm — MUStARD++

```bash
python MPP_Code/training/execute_sarcasm_mustard++.py -s n -m VTA -c y
```

Expects `MPP_Code/data/final_datasets/mustard++_sarcasm_detection.csv` and splits under `MPP_Code/data/split_mustard_pp_sarcasm/`.

### Sarcasm — original MUStARD

```bash
python MPP_Code/training/execute_sarcasm_mustard.py -s n -m VTA -c y
```

**Important:** this file contains a **hardcoded path** to `MUStARD-Final.csv` (a Linux path). Edit that `pd.read_csv(...)` line to point to your local CSV before running.

---

## 6. Quick checklist

1. `cd` to repository root.  
2. Set `PYTHONPATH` to `.../MPP_Code`.  
3. Install dependencies (`torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`).  
4. Ensure **CSV metadata + splits** (local) sit where your script expects, and **pickles** from `data/extracted_features/` match that script; fix paths if your layout differs.  
5. Create output directories under `MPP_Code/log/` (and related) if needed.  
6. Run the chosen `python MPP_Code/training/<script>.py ...` command.

---

## 7. Project layout (reference)

| Path | Role |
|------|------|
| `models/` | Model definitions (`emotion_classification_model.py`, `emotion_regression_model.py`) |
| `training/` | Executable training / tuning scripts |
| `data/` | Versioned: `extracted_features/*.pkl`. Local only: CSVs, splits, video folders (see §3) |

Features are not recomputed inside these training scripts; they load pre-extracted tensors from pickles as configured at the top of each training file.
