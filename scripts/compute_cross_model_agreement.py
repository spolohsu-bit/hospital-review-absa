#!/usr/bin/env python3
"""
Compute cross-model agreement metrics for JMIR #92325 R2, Comment #19.

Compares DeepSeek and Claude ABSA outputs against:
1. Human gold standard (consensus of Human1 + Human2)
2. GPT-4o original output

Reports: Cohen κ, accuracy, per-aspect F1 for each model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report

SCRIPT_DIR = Path(__file__).parent
GOLD_PATH = SCRIPT_DIR / "gold_standard.xlsx"
RESULTS_DIR = SCRIPT_DIR / "results"

ASPECTS = [
    "Administrative Processes",
    "Emergency Care",
    "Facility & Environment",
    "Inpatient Care",
    "Professional Quality",
    "Service Attitude",
    "Surgical & Specialty Care",
]


def load_gold():
    """Load gold standard: Human2 consensus as ground truth."""
    df = pd.read_excel(GOLD_PATH)
    df["Human2"] = pd.to_numeric(df["Human2"], errors="coerce")
    df["gpt"] = pd.to_numeric(df["gpt"], errors="coerce")
    return df


def discretize(score):
    if pd.isna(score):
        return np.nan
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def load_model_results(model_name: str) -> dict:
    """Load model results as {review_id: {aspect: score}}."""
    path = RESULTS_DIR / f"{model_name}_absa_results.json"
    if not path.exists():
        print(f"  {path} not found, skipping {model_name}")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["review_id"]: r["aspect_scores"] for r in data}


def build_long_format(model_results: dict, gold_df: pd.DataFrame) -> pd.DataFrame:
    """Merge model predictions with gold standard in long format."""
    rows = []
    for _, row in gold_df.iterrows():
        rid = int(row["review_id"])
        aspect = row["dimension"]
        if aspect not in ASPECTS:
            continue
        gold_val = row["Human2"]
        gpt_val = row["gpt"]
        model_scores = model_results.get(rid, {})
        model_val = model_scores.get(aspect, np.nan)

        rows.append({
            "review_id": rid,
            "aspect": aspect,
            "gold": discretize(gold_val),
            "gpt": discretize(gpt_val),
            "model": discretize(model_val),
        })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, pred_col: str, label: str):
    """Compute κ and accuracy for a prediction column vs gold."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 4-class (absent=-99, neg=-1, neu=0, pos=1)
    ABSENT = -99
    gold_4 = df["gold"].fillna(ABSENT).astype(int)
    pred_4 = df[pred_col].fillna(ABSENT).astype(int)

    kappa_4 = cohen_kappa_score(gold_4, pred_4)
    acc_4 = accuracy_score(gold_4, pred_4)
    print(f"  4-class (absent/neg/neu/pos): κ={kappa_4:.3f}, acc={acc_4:.3f}")

    # Aspect detection: present vs absent
    gold_present = (~df["gold"].isna()).astype(int)
    pred_present = (~df[pred_col].isna()).astype(int)
    kappa_det = cohen_kappa_score(gold_present, pred_present)
    acc_det = accuracy_score(gold_present, pred_present)
    print(f"  Aspect detection (binary): κ={kappa_det:.3f}, acc={acc_det:.3f}")

    # Per-aspect detection F1
    print(f"\n  Per-aspect detection F1:")
    for aspect in ASPECTS:
        mask = df["aspect"] == aspect
        g = gold_present[mask]
        p = pred_present[mask]
        f1 = f1_score(g, p, zero_division=0)
        prec = (g[p == 1] == 1).mean() if (p == 1).sum() > 0 else 0
        rec = (p[g == 1] == 1).mean() if (g == 1).sum() > 0 else 0
        print(f"    {aspect:<30} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    # Sentiment agreement (conditional on both detecting)
    both = df.dropna(subset=["gold", pred_col])
    if len(both) > 0:
        kappa_sent = cohen_kappa_score(both["gold"].astype(int), both[pred_col].astype(int))
        acc_sent = accuracy_score(both["gold"].astype(int), both[pred_col].astype(int))
        print(f"\n  Sentiment (conditional): κ={kappa_sent:.3f}, acc={acc_sent:.3f}, n={len(both)}")
    else:
        print(f"\n  Sentiment: no overlapping detections")

    return {
        "label": label,
        "kappa_4class": kappa_4,
        "acc_4class": acc_4,
        "kappa_detection": kappa_det,
        "acc_detection": acc_det,
        "kappa_sentiment": kappa_sent if len(both) > 0 else np.nan,
        "n_sentiment": len(both),
    }


def compute_intermodel(df: pd.DataFrame, col_a: str, col_b: str, label: str):
    """Compute agreement between two model columns."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    ABSENT = -99
    a_4 = df[col_a].fillna(ABSENT).astype(int)
    b_4 = df[col_b].fillna(ABSENT).astype(int)
    kappa = cohen_kappa_score(a_4, b_4)
    acc = accuracy_score(a_4, b_4)
    print(f"  4-class agreement: κ={kappa:.3f}, acc={acc:.3f}")

    both = df.dropna(subset=[col_a, col_b])
    if len(both) > 0:
        kappa_sent = cohen_kappa_score(both[col_a].astype(int), both[col_b].astype(int))
        print(f"  Sentiment (conditional): κ={kappa_sent:.3f}, n={len(both)}")

    return {"label": label, "kappa_4class": kappa, "acc_4class": acc}


def main():
    gold_df = load_gold()
    print(f"Gold standard: {gold_df['review_id'].nunique()} reviews, {len(gold_df)} rows")

    summary = []

    for model_name in ["deepseek", "claude"]:
        model_results = load_model_results(model_name)
        if not model_results:
            continue

        print(f"\nLoaded {len(model_results)} {model_name} results")
        df = build_long_format(model_results, gold_df)

        # Model vs Gold
        metrics = compute_metrics(df, "model", f"{model_name.upper()} vs Gold Standard")
        summary.append(metrics)

        # GPT vs Gold (for comparison)
        if model_name == "deepseek":  # only compute once
            gpt_metrics = compute_metrics(df, "gpt", "GPT-4o vs Gold Standard (reference)")
            summary.append(gpt_metrics)

        # Model vs GPT (inter-model)
        im = compute_intermodel(df, "model", "gpt", f"{model_name.upper()} vs GPT-4o")
        summary.append(im)

    # Save summary
    if summary:
        out_path = RESULTS_DIR / "cross_model_summary.csv"
        pd.DataFrame(summary).to_csv(out_path, index=False)
        print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
