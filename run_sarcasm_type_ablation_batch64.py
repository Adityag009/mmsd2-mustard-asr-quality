#!/usr/bin/env python3
"""
Run the batch-64 sarcasm-type ablation experiment from inside the
mmsd2-mustard-asr-quality repository.

Place this file anywhere inside the repository, for example:

    experiments/run_sarcasm_type_ablation_batch64.py

Then run from the repository root:

    python experiments/run_sarcasm_type_ablation_batch64.py

The script also works when launched from another directory because it can infer
or accept the repository root with --repo-dir.

What it does:
1. Finds the repository root and configures PYTHONPATH=<repo>/MPP_Code.
2. Builds MPP_Code/data/final_datasets/mustard++_sarcasm_detection.csv.
3. Builds 5 binary-stratified splits in MPP_Code/data/split_mustard_pp_sarcasm.
4. Creates required log/chart/stats/prediction directories.
5. Creates a patched copy of MPP_Code/training/execute_sarcasm_mustard++.py so
   prediction CSVs include SAR_T, without editing tracked training code.
6. Runs modalities T, A, V, TA, VA, VT, VTA with and without context.
7. Copies each run's newest prediction CSV into sarcasm_type_ablation_results/.
8. Computes type-conditioned and overall binary metric CSV summaries.

Common commands:

    # Prepare data/splits and check paths only.
    python experiments/run_sarcasm_type_ablation_batch64.py --prepare-only

    # Run only the final modalities used in the paragraph/table.
    python experiments/run_sarcasm_type_ablation_batch64.py --modalities VT VTA --contexts y

    # Summarize existing copied predictions without retraining.
    python experiments/run_sarcasm_type_ablation_batch64.py --skip-training

    # Print training commands without running them.
    python experiments/run_sarcasm_type_ablation_batch64.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


DEFAULT_MODALITIES = ["T", "A", "V", "TA", "VA", "VT", "VTA"]
DEFAULT_CONTEXTS = ["n", "y"]

DEFAULT_CONFIG = {
    "epochs": "500",
    "batch_size": "64",
    "learning_rate": "0.001",
    "patience": "5",
    "dropout": "0.3",
    "shared": "1024",
    "projection": "256",
    "classification_report": "y",
    "gpu": "0",
    "speaker": "n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the batch-64 sarcasm-type ablation experiment from inside the git repository."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=None,
        help=(
            "Repository root. If omitted, the script searches upward from the script location "
            "and current working directory."
        ),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help=(
            "MUSTARD++ text/annotation CSV. Default search order: "
            "<repo>/mustard++_text.csv, then <repo>/MPP_Code/data/mustard_PP_utterance.csv."
        ),
    )
    parser.add_argument(
        "--training-script",
        type=Path,
        default=None,
        help="Training script. Default: <repo>/MPP_Code/training/execute_sarcasm_mustard++.py.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory. Default: <repo>/sarcasm_type_ablation_results.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=DEFAULT_MODALITIES,
        help="Modalities to run. Default: T A V TA VA VT VTA.",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=DEFAULT_CONTEXTS,
        choices=["n", "y"],
        help="Contexts to run: n=without context, y=with context. Default: n y.",
    )
    parser.add_argument("--epochs", default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size", default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--patience", default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--dropout", default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--shared", default=DEFAULT_CONFIG["shared"])
    parser.add_argument("--projection", default=DEFAULT_CONFIG["projection"])
    parser.add_argument("--classification-report", default=DEFAULT_CONFIG["classification_report"])
    parser.add_argument("--gpu", default=DEFAULT_CONFIG["gpu"])
    parser.add_argument("--speaker", default=DEFAULT_CONFIG["speaker"])
    parser.add_argument(
        "--patch-mode",
        choices=["copy", "inplace", "none"],
        default="copy",
        help=(
            "How to make prediction CSVs include SAR_T. "
            "copy creates and runs an untracked patched copy in the results directory; "
            "inplace edits the original training script after creating a .bak; "
            "none uses the training script as-is. Default: copy."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create data/splits/directories and patched training copy, then stop before training.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and summarize existing copied prediction CSVs from the results directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, but do not start training or compute summaries.",
    )
    return parser.parse_args()


def is_repo_root(path: Path) -> bool:
    return (
        (path / "MPP_Code" / "training").is_dir()
        and (path / "MPP_Code" / "models").is_dir()
        and ((path / "README.md").exists() or (path / "pyproject.toml").exists())
    )


def find_repo_root(explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([Path(__file__).resolve().parent, Path.cwd().resolve()])

    checked: List[Path] = []
    for start in candidates:
        start = start.expanduser().resolve()
        walk = [start] + list(start.parents)
        for candidate in walk:
            if candidate in checked:
                continue
            checked.append(candidate)
            if is_repo_root(candidate):
                return candidate

    message = "Could not infer the repository root. Pass --repo-dir explicitly. Checked:\n"
    message += "\n".join(f"- {p}" for p in checked[:30])
    raise FileNotFoundError(message)


def resolve_source_csv(repo_dir: Path, source_csv: Optional[Path]) -> Path:
    if source_csv is not None:
        source_csv = source_csv.expanduser().resolve()
        if not source_csv.exists():
            raise FileNotFoundError(f"Source CSV does not exist: {source_csv}")
        return source_csv

    candidates = [
        repo_dir / "mustard++_text.csv",
        repo_dir / "MPP_Code" / "data" / "mustard_PP_utterance.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the source CSV. Expected one of:\n"
        + "\n".join(f"- {p}" for p in candidates)
        + "\nPass --source-csv if your file is elsewhere."
    )


def resolve_training_script(repo_dir: Path, training_script: Optional[Path]) -> Path:
    if training_script is not None:
        path = training_script.expanduser().resolve()
    else:
        path = repo_dir / "MPP_Code" / "training" / "execute_sarcasm_mustard++.py"
    if not path.exists():
        raise FileNotFoundError(f"Training script does not exist: {path}")
    return path


def configure_pythonpath(repo_dir: Path) -> None:
    """Mirror the repository's documented training setup."""
    mpp_code = repo_dir / "MPP_Code"
    existing = os.environ.get("PYTHONPATH", "")
    paths = [str(mpp_code)]
    if existing:
        paths.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    if str(mpp_code) not in sys.path:
        sys.path.insert(0, str(mpp_code))


def build_config(args: argparse.Namespace) -> Dict[str, str]:
    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "learning_rate": str(args.learning_rate),
            "patience": str(args.patience),
            "dropout": str(args.dropout),
            "shared": str(args.shared),
            "projection": str(args.projection),
            "classification_report": str(args.classification_report),
            "gpu": str(args.gpu),
            "speaker": str(args.speaker),
        }
    )
    return config


def build_sarcasm_type_dataset(repo_dir: Path, source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv, dtype={"SCENE": str})

    required_columns = {"SCENE", "SPEAKER", "Sarcasm"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {sorted(missing)}")

    # Keep utterance rows and remove context rows when the KEY convention is present.
    if "KEY" in df.columns:
        df = df[~df["KEY"].astype(str).str.contains(r"_c_\d+$", regex=True)].copy()

    df = df[df["Sarcasm"].notna()].copy()
    df["SAR"] = df["Sarcasm"].astype(int)

    if "Sarcasm_Type" in df.columns:
        df["SAR_T"] = df["Sarcasm_Type"].fillna("NON_SARCASM").astype(str)
    else:
        print("Warning: source CSV has no Sarcasm_Type column; sarcastic rows get SAR_T='UNKNOWN'.")
        df["SAR_T"] = "UNKNOWN"
    df.loc[df["SAR"] == 0, "SAR_T"] = "NON_SARCASM"

    sarcasm_df = df[["SCENE", "SPEAKER", "SAR", "SAR_T"]].copy()

    final_dir = repo_dir / "MPP_Code" / "data" / "final_datasets"
    split_dir = repo_dir / "MPP_Code" / "data" / "split_mustard_pp_sarcasm"
    final_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    sarcasm_path = final_dir / "mustard++_sarcasm_detection.csv"
    sarcasm_df.to_csv(sarcasm_path, index=False)

    print(f"Wrote dataset: {sarcasm_path}")
    print("Binary SAR distribution:")
    print(sarcasm_df["SAR"].value_counts().sort_index())
    print("Sarcasm type distribution:")
    print(sarcasm_df["SAR_T"].value_counts())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(sarcasm_df["SCENE"], sarcasm_df["SAR"])):
        train = sarcasm_df.iloc[train_idx].reset_index(drop=True)
        test = sarcasm_df.iloc[test_idx].reset_index(drop=True)
        train.to_csv(split_dir / f"train_{fold}.csv", index=False)
        test.to_csv(split_dir / f"test_{fold}.csv", index=False)
        print(f"Fold {fold}: train={train.shape}, test={test.shape}")

    return sarcasm_df


def ensure_output_directories(repo_dir: Path, modalities: Iterable[str]) -> None:
    folders = [
        "MPP_Code/log/sarcasm",
        "MPP_Code/charts/sarcasm",
        "MPP_Code/stats/sarcasm",
        "MPP_Code/saved_models/sarc",
    ]
    folders.extend(f"MPP_Code/predictions/sarcasm/{modality}" for modality in modalities)

    for folder in folders:
        (repo_dir / folder).mkdir(parents=True, exist_ok=True)


def make_results_dirs(repo_dir: Path, results_dir: Optional[Path]) -> Dict[str, Path]:
    root = results_dir.expanduser().resolve() if results_dir is not None else repo_dir / "sarcasm_type_ablation_results"
    dirs = {
        "root": root,
        "logs": root / "logs",
        "predictions": root / "predictions",
        "summary": root / "summary",
        "patched": root / "_patched_training",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def patch_training_text(text: str) -> str:
    patched = text.replace(
        "# types.extend(valid['SAR_T'].tolist())",
        "types.extend(valid['SAR_T'].tolist())",
    )

    old_block = (
        "for row in zip(indexes, true_all, pred_all):\n"
        "    results.append(row)\n\n"
        "results = pd.DataFrame(\n"
        "    results, columns=['KEY', 'TRUE', 'PRED'])"
    )
    new_block = (
        "for row in zip(indexes, true_all, pred_all, types):\n"
        "    results.append(row)\n\n"
        "results = pd.DataFrame(\n"
        "    results, columns=['KEY', 'TRUE', 'PRED', 'SAR_T'])"
    )
    if old_block in patched:
        patched = patched.replace(old_block, new_block)

    return patched


def prepare_training_script(
    original_script: Path,
    patch_mode: str,
    patched_dir: Path,
) -> Path:
    if patch_mode == "none":
        return original_script

    text = original_script.read_text(encoding="utf-8", errors="ignore")
    patched = patch_training_text(text)

    if "columns=['KEY', 'TRUE', 'PRED', 'SAR_T']" not in patched:
        print(
            "Warning: the SAR_T prediction-output patch was not detected. "
            "Prediction summaries requiring SAR_T may fail unless the training script already saves it."
        )

    if patch_mode == "inplace":
        backup = original_script.with_suffix(".py.sarcasm_type.bak")
        if not backup.exists():
            backup.write_text(text, encoding="utf-8")
            print(f"Backup created: {backup}")
        if patched != text:
            original_script.write_text(patched, encoding="utf-8")
            print(f"Patched in place: {original_script}")
        else:
            print("Training script already appears patched; no in-place changes written.")
        return original_script

    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_script = patched_dir / original_script.name
    patched_script.write_text(patched, encoding="utf-8")
    print(f"Using patched training copy: {patched_script}")
    return patched_script


def command_for_run(
    python_executable: str,
    training_script: Path,
    modality: str,
    context: str,
    config: Dict[str, str],
) -> List[str]:
    return [
        python_executable,
        str(training_script),
        "-s",
        config["speaker"],
        "-m",
        modality,
        "-c",
        context,
        "-e",
        config["epochs"],
        "-b",
        config["batch_size"],
        "-l",
        config["learning_rate"],
        "-p",
        config["patience"],
        "-d",
        config["dropout"],
        "-sh",
        config["shared"],
        "-pr",
        config["projection"],
        "-cr",
        config["classification_report"],
        "-gpu",
        config["gpu"],
    ]


def copy_newest_prediction(repo_dir: Path, modality: str, destination: Path) -> Optional[Path]:
    repo_pred_dir = repo_dir / "MPP_Code" / "predictions" / "sarcasm" / modality
    pred_files = sorted(repo_pred_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pred_files:
        print(f"Warning: no prediction CSV found in {repo_pred_dir}")
        return None
    shutil.copy(pred_files[0], destination)
    return destination


def run_one_ablation(
    repo_dir: Path,
    training_script: Path,
    results_dirs: Dict[str, Path],
    modality: str,
    context: str,
    config: Dict[str, str],
) -> Dict[str, Any]:
    cmd = command_for_run(sys.executable, training_script, modality, context, config)
    run_name = f"{modality}_context_{context}"

    print("\n" + "=" * 100)
    print(f"Running {run_name}")
    print(" ".join(cmd))
    print("=" * 100)

    start = datetime.now()
    result = subprocess.run(
        cmd,
        cwd=repo_dir,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    runtime_seconds = (datetime.now() - start).total_seconds()

    stdout_path = results_dirs["logs"] / f"{run_name}_stdout_batch64.txt"
    stderr_path = results_dirs["logs"] / f"{run_name}_stderr_batch64.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    print(f"Return code: {result.returncode}")
    print(f"Runtime seconds: {runtime_seconds:.2f}")
    print(f"Saved stdout: {stdout_path}")
    print(f"Saved stderr: {stderr_path}")

    copied_pred: Optional[Path] = None
    if result.returncode == 0:
        destination = results_dirs["predictions"] / f"{run_name}_predictions.csv"
        copied_pred = copy_newest_prediction(repo_dir, modality, destination)
        if copied_pred:
            print(f"Copied predictions: {copied_pred}")
    else:
        print("Last stderr lines:")
        print(result.stderr[-4000:])

    return {
        "modality": modality,
        "context": context,
        "returncode": result.returncode,
        "runtime_seconds": runtime_seconds,
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "prediction_file": str(copied_pred) if copied_pred else None,
        "command": cmd,
    }


def runs_from_existing_predictions(preds_dir: Path, modalities: Iterable[str], contexts: Iterable[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for context in contexts:
        for modality in modalities:
            run_name = f"{modality}_context_{context}"
            pred_file = preds_dir / f"{run_name}_predictions.csv"
            runs.append(
                {
                    "modality": modality,
                    "context": context,
                    "returncode": 0 if pred_file.exists() else 1,
                    "prediction_file": str(pred_file) if pred_file.exists() else None,
                }
            )
    return runs


def validate_prediction_columns(pred_df: pd.DataFrame, pred_path: Path, require_sar_t: bool = True) -> None:
    required = {"TRUE", "PRED"}
    if require_sar_t:
        required.add("SAR_T")
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction file {pred_path} is missing required columns: {sorted(missing)}")


def compute_type_conditioned_f1(runs: Sequence[Dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("returncode") != 0 or not run.get("prediction_file"):
            continue
        pred_path = Path(run["prediction_file"])
        pred_df = pd.read_csv(pred_path)
        validate_prediction_columns(pred_df, pred_path, require_sar_t=True)
        pred_df["TRUE"] = pred_df["TRUE"].astype(int)
        pred_df["PRED"] = pred_df["PRED"].astype(int)

        sarcasm_types = sorted(
            t
            for t in pred_df["SAR_T"].dropna().unique()
            if t != "NON_SARCASM" and str(t).lower() != "none"
        )
        for sar_type in sarcasm_types:
            pos = pred_df[(pred_df["TRUE"] == 1) & (pred_df["SAR_T"] == sar_type)]
            neg = pred_df[pred_df["TRUE"] == 0]
            subset = pd.concat([pos, neg], ignore_index=True)
            if len(pos) == 0 or subset["TRUE"].nunique() < 2:
                continue
            y_true = subset["TRUE"]
            y_pred = subset["PRED"]
            rows.append(
                {
                    "modality": run["modality"],
                    "context": "with_context" if run["context"] == "y" else "without_context",
                    "sarcasm_type": sar_type,
                    "positive_support": len(pos),
                    "negative_support": len(neg),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision_for_sarcasm": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "recall_for_sarcasm_type": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "f1_for_sarcasm_type": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = summary_dir / "sarcasm_type_conditioned_f1_scores_batch64.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if not df.empty:
        pivot = df.pivot_table(
            index=["sarcasm_type", "context"],
            columns="modality",
            values="f1_for_sarcasm_type",
            aggfunc="mean",
        ).reset_index()
        pivot_csv = summary_dir / "sarcasm_type_conditioned_f1_pivot_batch64.csv"
        pivot.to_csv(pivot_csv, index=False)
        print(f"Saved: {pivot_csv}")
    return df


def compute_type_detection_recall(runs: Sequence[Dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("returncode") != 0 or not run.get("prediction_file"):
            continue
        pred_path = Path(run["prediction_file"])
        pred_df = pd.read_csv(pred_path)
        validate_prediction_columns(pred_df, pred_path, require_sar_t=True)
        pred_df["TRUE"] = pred_df["TRUE"].astype(int)
        pred_df["PRED"] = pred_df["PRED"].astype(int)

        sar_df = pred_df[pred_df["TRUE"] == 1].copy()
        for sar_type, group in sar_df.groupby("SAR_T"):
            if sar_type == "NON_SARCASM" or str(sar_type).lower() == "none":
                continue
            rows.append(
                {
                    "modality": run["modality"],
                    "context": "with_context" if run["context"] == "y" else "without_context",
                    "sarcasm_type": sar_type,
                    "support": len(group),
                    "type_detection_recall": (group["PRED"] == 1).mean(),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = summary_dir / "sarcasm_type_detection_recall_batch64.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if not df.empty:
        pivot = df.pivot_table(
            index=["sarcasm_type", "context"],
            columns="modality",
            values="type_detection_recall",
            aggfunc="mean",
        ).reset_index()
        pivot_csv = summary_dir / "sarcasm_type_detection_recall_pivot_batch64.csv"
        pivot.to_csv(pivot_csv, index=False)
        print(f"Saved: {pivot_csv}")
    return df


def compute_overall_binary_metrics(runs: Sequence[Dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("returncode") != 0 or not run.get("prediction_file"):
            continue
        pred_path = Path(run["prediction_file"])
        pred_df = pd.read_csv(pred_path)
        validate_prediction_columns(pred_df, pred_path, require_sar_t=False)

        y_true = pred_df["TRUE"].astype(int)
        y_pred = pred_df["PRED"].astype(int)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )

        rows.append(
            {
                "modality": run["modality"],
                "context": "with_context" if run["context"] == "y" else "without_context",
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "macro_f1": f1_macro,
                "precision_sarcasm": precision_binary,
                "recall_sarcasm": recall_binary,
                "f1_sarcasm": f1_binary,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = summary_dir / "overall_binary_metrics_batch64.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if not df.empty:
        pivot = df.pivot_table(index="modality", columns="context", values="macro_f1", aggfunc="mean").reset_index()
        if {"with_context", "without_context"}.issubset(pivot.columns):
            pivot["context_effect"] = pivot["with_context"] - pivot["without_context"]
        pivot_csv = summary_dir / "overall_macro_f1_pivot_batch64.csv"
        pivot.to_csv(pivot_csv, index=False)
        print(f"Saved: {pivot_csv}")
    return df


def print_path_summary(repo_dir: Path, source_csv: Path, training_script: Path, results_dirs: Dict[str, Path]) -> None:
    print(f"Repository root: {repo_dir}")
    print(f"Source CSV:      {source_csv}")
    print(f"Training script: {training_script}")
    print(f"Results dir:     {results_dirs['root']}")
    print(f"PYTHONPATH head: {(repo_dir / 'MPP_Code')}")


def main() -> None:
    args = parse_args()
    repo_dir = find_repo_root(args.repo_dir)
    source_csv = resolve_source_csv(repo_dir, args.source_csv)
    original_training_script = resolve_training_script(repo_dir, args.training_script)
    results_dirs = make_results_dirs(repo_dir, args.results_dir)

    configure_pythonpath(repo_dir)
    training_script = prepare_training_script(original_training_script, args.patch_mode, results_dirs["patched"])
    config = build_config(args)

    print_path_summary(repo_dir, source_csv, training_script, results_dirs)
    build_sarcasm_type_dataset(repo_dir, source_csv)
    ensure_output_directories(repo_dir, args.modalities)

    if args.prepare_only:
        print("Preparation complete. Stopping because --prepare-only was set.")
        return

    if args.dry_run:
        print("Dry run commands:")
        for context in args.contexts:
            for modality in args.modalities:
                cmd = command_for_run(sys.executable, training_script, modality, context, config)
                print(" ".join(cmd))
        return

    if args.skip_training:
        print("Skipping training. Reading existing copied prediction CSVs from results directory.")
        runs = runs_from_existing_predictions(results_dirs["predictions"], args.modalities, args.contexts)
    else:
        runs: List[Dict[str, Any]] = []
        for context in args.contexts:
            for modality in args.modalities:
                runs.append(
                    run_one_ablation(
                        repo_dir=repo_dir,
                        training_script=training_script,
                        results_dirs=results_dirs,
                        modality=modality,
                        context=context,
                        config=config,
                    )
                )

    metadata_path = results_dirs["summary"] / "sarcasm_type_ablation_run_metadata_batch64.json"
    metadata_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")

    compute_type_conditioned_f1(runs, results_dirs["summary"])
    compute_type_detection_recall(runs, results_dirs["summary"])
    compute_overall_binary_metrics(runs, results_dirs["summary"])

    print("Done.")


if __name__ == "__main__":
    main()
