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

That creates or updates `.venv` and installs the same dependencies from `pyproject.toml` (a `uv.lock` file may be generated for reproducible installs).

The scripts assume a **CUDA GPU** by default (`cuda:0`, `cuda:1`, etc.). To train on CPU you would need to edit each script and change `torch.device("cuda:...")` to `torch.device("cpu")`.

---

## 2. Working directory and Python path

All commands below assume your **current working directory is the repository root** (the folder that contains `MPP_Code/`), because paths like `MPP_Code/data/...` are written relative to that root.

Training scripts use `from models import ...`, so **`MPP_Code` must be on `PYTHONPATH`**.

**Windows (PowerShell), from the repo root:**

```powershell
cd path\to\MUStARD_Plus_Plus
$env:PYTHONPATH = "$PWD\MPP_Code"
```

**Linux / macOS:**

```bash
cd /path/to/MUStARD_Plus_Plus
export PYTHONPATH="$(pwd)/MPP_Code"
```

---

## 3. Data you need

The repo does **not** ship full pre-extracted features. You typically need:

- CSVs and **train/test split** files under `MPP_Code/data/` (exact filenames differ per script; see section 5).
- **Pickle files** of extracted multimodal features under `MPP_Code/data/extracted_features/` (and subfolders such as `an_merged/`), matching what each script opens.

Obtain videos and align feature extraction with the paper / original authors’ setup. The root README links to the [Google Drive folder](https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW?usp=sharing) for video assets.

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
4. Place CSVs, pickles, and split files where each script expects them (or update paths in the script).  
5. Create output directories under `MPP_Code/log/` (and related) if needed.  
6. Run the chosen `python MPP_Code/training/<script>.py ...` command.

---

## 7. Project layout (reference)

| Path | Role |
|------|------|
| `models/` | Model definitions (`emotion_classification_model.py`, `emotion_regression_model.py`) |
| `training/` | Executable training / tuning scripts |
| `data/` | You populate: CSVs, `extracted_features/`, split CSVs |

Features are not recomputed inside these training scripts; they load pre-extracted tensors from pickles as configured at the top of each training file.
