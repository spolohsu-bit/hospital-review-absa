#!/usr/bin/env python3
"""Compute IRR (Human1 vs Human2) and GPT accuracy vs Human2.

Designed for `hospital_gold_long_with_gpt*.xlsx` files.

Assumptions:
- Human1/Human2 are sentiment labels in {-1, 0, +1}; missing means aspect not present.
- GPT column stores an average sentiment score (may be fractional). We discretize by sign:
    score > 0 -> +1
    score < 0 -> -1
    score == 0 -> 0

Metrics:
- IRR:
    * sentiment IRR on rows where both humans labeled (Cohen's kappa + quadratic-weighted kappa).
    * if `--all-rows` is set, also report IRR on the full grid treating NaN as \"absent\" (a 4th class),
      and presence-only IRR (absent vs present).
- GPT accuracy vs Human2 (gold):
    * sentiment accuracy on rows where both GPT and Human2 have labels.
    * if `--all-rows` is set, also report end-to-end accuracy on the full grid treating NaN as \"absent\"
      (a 4th class), plus aspect-presence precision/recall/F1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


LABELS = [-1, 0, 1]
ABSENT = 2  # 4th class used only in --all-rows mode; distinct from neutral (0).


def clean_numeric(series: pd.Series) -> pd.Series:
    # Treat whitespace-only strings and '-' as missing.
    s = series.replace(r"^\s*$", np.nan, regex=True).replace("-", np.nan)
    return pd.to_numeric(s, errors="coerce")


def gpt_to_label(x):
    if pd.isna(x):
        return np.nan
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def to_4class(x):
    # Map missing -> ABSENT, keep {-1,0,1} unchanged.
    return ABSENT if pd.isna(x) else int(x)


def confusion_matrix(gold: Iterable[int], pred: Iterable[int], labels: List[int]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gold, pred):
        cm[idx[g], idx[p]] += 1
    return cm


def kappa_from_cm(cm: np.ndarray) -> Tuple[float, float, float]:
    n = cm.sum()
    po = float(np.trace(cm) / n) if n else float("nan")
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    pe = float(np.dot(row, col))
    kappa = 1.0 if pe == 1 else float((po - pe) / (1 - pe))
    return kappa, po, pe


def classification_metrics_from_cm(cm: np.ndarray, labels: List[int]) -> dict:
    kappa, po, pe = kappa_from_cm(cm)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")

    per = {}
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
        per[lab] = {"support": support, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    macro_prec = float(np.nanmean([per[lab]["precision"] for lab in labels]))
    macro_rec = float(np.nanmean([per[lab]["recall"] for lab in labels]))
    macro_f1 = float(np.nanmean([per[lab]["f1"] for lab in labels]))

    supports = np.array([per[lab]["support"] for lab in labels], dtype=float)

    def weighted_avg(metric: str) -> float:
        vals = np.array([per[lab][metric] for lab in labels], dtype=float)
        w = supports.copy()
        w[np.isnan(vals)] = 0.0
        denom = w.sum()
        return float(np.nansum(w * vals) / denom) if denom else float("nan")

    weighted_prec = weighted_avg("precision")
    weighted_rec = weighted_avg("recall")
    weighted_f1 = weighted_avg("f1")

    return {
        "accuracy": acc,
        "kappa": kappa,
        "agreement": po,
        "expected_agreement": pe,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "per_class": per,
    }


def cohen_kappa(a_vals: Iterable[int], b_vals: Iterable[int], labels: List[int]) -> Tuple[float, float, float]:
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(a_vals, b_vals):
        mat[lab_to_idx[a], lab_to_idx[b]] += 1

    n = mat.sum()
    po = float(np.trace(mat) / n) if n else float("nan")
    row = mat.sum(axis=1) / n
    col = mat.sum(axis=0) / n
    pe = float(np.dot(row, col))
    kappa = 1.0 if pe == 1 else float((po - pe) / (1 - pe))
    return kappa, po, pe


def weighted_kappa(a_vals: Iterable[int], b_vals: Iterable[int], labels: List[int], weight: str = "quadratic") -> float:
    k = len(labels)
    if k < 2:
        return float("nan")

    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((k, k), dtype=float)
    for a, b in zip(a_vals, b_vals):
        mat[lab_to_idx[a], lab_to_idx[b]] += 1

    n = mat.sum()
    o = mat / n
    row = o.sum(axis=1)
    col = o.sum(axis=0)
    e = np.outer(row, col)

    w = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weight == "linear":
                w[i, j] = abs(i - j) / (k - 1)
            else:
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    num = float((w * o).sum())
    den = float((w * e).sum())
    return float("nan") if den == 0 else float(1 - (num / den))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default="Hospital Gold Long Format")
    ap.add_argument("--human1-col", default="Human1")
    ap.add_argument("--human2-col", default="Human2")
    ap.add_argument("--gpt-col", default="gpt")
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="Compute full-grid metrics using 4-class labels (-1/0/+1/absent), where NaN means absent (NOT neutral=0).",
    )
    ap.add_argument(
        "--exclude-dimension",
        action="append",
        default=[],
        help="Exclude rows where `dimension` matches this value (can be repeated).",
    )
    args = ap.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)

    if args.exclude_dimension:
        # Handle minor singular/plural variations (eg, Medical Cost vs Medical Costs).
        exclude = set()
        for d in args.exclude_dimension:
            if not d:
                continue
            d = str(d).strip()
            if not d:
                continue
            exclude.add(d)
            if d.endswith("s"):
                exclude.add(d[:-1])
            else:
                exclude.add(d + "s")

        before = len(df)
        df = df[~df["dimension"].astype(str).isin(exclude)].copy()
        dropped = before - len(df)
        print(f"Excluded dimensions={sorted(exclude)}; dropped_rows={dropped}")

    h1 = clean_numeric(df[args.human1_col])
    h2 = clean_numeric(df[args.human2_col])
    gpt_raw = pd.to_numeric(df[args.gpt_col], errors="coerce")
    gpt_lab = gpt_raw.apply(gpt_to_label)

    if args.all_rows:
        labels_4 = [-1, 0, 1, ABSENT]

        # Full-grid metrics treat NaN as a distinct "absent" class (NOT neutral=0).
        h1_4 = h1.apply(to_4class).tolist()
        h2_4 = h2.apply(to_4class).tolist()

        cm_irr_4 = confusion_matrix(h2_4, h1_4, labels_4)  # rows=Human2, cols=Human1
        irr_4 = classification_metrics_from_cm(cm_irr_4, labels_4)
        print("IRR (Human1 vs Human2, all rows; NaN=absent)")
        print(f"- n={len(df)}")
        print(f"- agreement={irr_4['agreement']:.4f}")
        print(f"- Cohen's kappa (4-class: -1/0/+1/absent)={irr_4['kappa']:.4f}")
        print(f"- confusion_matrix rows=Human2 cols=Human1 labels [-1,0,1,absent]: {cm_irr_4.tolist()}")

        # Presence-only IRR (absent vs present)
        h1_present = (h1.notna()).astype(int).tolist()
        h2_present = (h2.notna()).astype(int).tolist()
        kappa_p, po_p, _ = cohen_kappa(h1_present, h2_present, [0, 1])
        print(f"- presence agreement={po_p:.4f}")
        print(f"- presence Cohen's kappa={kappa_p:.4f}")

        # Sentiment IRR on rows where both present
        mask_sent = h1.notna() & h2.notna()
        a_sent = h1[mask_sent].astype(int).tolist()
        b_sent = h2[mask_sent].astype(int).tolist()
        kappa_s, po_s, _ = cohen_kappa(a_sent, b_sent, LABELS)
        wk_s = weighted_kappa(a_sent, b_sent, LABELS, weight="quadratic")
        print("\nIRR (Human1 vs Human2, sentiment; both present)")
        print(f"- n={len(a_sent)}")
        print(f"- agreement={po_s:.4f}")
        print(f"- Cohen's kappa={kappa_s:.4f}")
        print(f"- weighted kappa (quadratic)={wk_s:.4f}")

        # GPT vs Human2: 4-class accuracy over full grid
        gold_4 = h2.apply(to_4class).tolist()
        pred_4 = gpt_lab.apply(to_4class).tolist()
        cm4 = confusion_matrix(gold_4, pred_4, labels_4)
        gpt_4m = classification_metrics_from_cm(cm4, labels_4)

        print("\nGPT accuracy (vs Human2 gold, all rows; NaN=absent)")
        print("- GPT discretization: score>0 -> +1, score<0 -> -1, score==0 -> 0, NaN -> absent")
        print(f"- n={len(df)}")
        print(f"- accuracy (4-class)={gpt_4m['accuracy']:.4f}")
        print(f"- Cohen's kappa (4-class)={gpt_4m['kappa']:.4f}")
        print(f"- macro_precision/recall/f1={gpt_4m['macro_precision']:.4f}/{gpt_4m['macro_recall']:.4f}/{gpt_4m['macro_f1']:.4f}")
        print(f"- weighted_precision/recall/f1={gpt_4m['weighted_precision']:.4f}/{gpt_4m['weighted_recall']:.4f}/{gpt_4m['weighted_f1']:.4f}")
        print(f"- confusion_matrix rows=gold cols=pred labels [-1,0,1,absent]: {cm4.tolist()}")
        print("- per-class metrics (label: support, precision, recall, f1)")
        for lab in labels_4:
            m = gpt_4m["per_class"][lab]
            lab_name = "不存在" if lab == ABSENT else str(lab)
            print(f"  {lab_name}: n={m['support']}, p={m['precision']:.4f}, r={m['recall']:.4f}, f1={m['f1']:.4f}")

        # Presence detection metrics
        gold_present = h2.notna().astype(int)
        pred_present = gpt_lab.notna().astype(int)
        tp = int(((pred_present == 1) & (gold_present == 1)).sum())
        fp = int(((pred_present == 1) & (gold_present == 0)).sum())
        tn = int(((pred_present == 0) & (gold_present == 0)).sum())
        fn = int(((pred_present == 0) & (gold_present == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
        acc_p = float((tp + tn) / len(df))
        print("\nGPT presence (aspect detection; present vs absent)")
        print(f"- acc={acc_p:.4f}")
        print(f"- precision={precision:.4f}")
        print(f"- recall={recall:.4f}")
        print(f"- f1={f1:.4f}")
        print(f"- TP/FP/TN/FN={tp}/{fp}/{tn}/{fn}")

        # Sentiment accuracy where both present (GPT and gold)
        mask_eval = gpt_lab.notna() & h2.notna()
        n_eval = int(mask_eval.sum())
        if n_eval:
            acc_sent = float(np.mean(gpt_lab[mask_eval].astype(int).values == h2[mask_eval].astype(int).values))
            print("\nGPT sentiment accuracy (both present)")
            print(f"- n={n_eval}")
            print(f"- accuracy={acc_sent:.4f}")
        return 0
    else:
        # IRR (sentiment) on overlapping labeled rows only
        mask_irr = h1.notna() & h2.notna()
        a = h1[mask_irr].astype(int).tolist()
        b = h2[mask_irr].astype(int).tolist()

        kappa, po, pe = cohen_kappa(a, b, LABELS)
        wk = weighted_kappa(a, b, LABELS, weight="quadratic")

        print("IRR (Human1 vs Human2, sentiment -1/0/+1)")
        print(f"- n={len(a)}")
        print(f"- agreement={po:.4f}")
        print(f"- Cohen's kappa={kappa:.4f}")
        print(f"- weighted kappa (quadratic)={wk:.4f}")
        print(f"- Human1 dist={Counter(a)}")
        print(f"- Human2 dist={Counter(b)}")

        # GPT accuracy vs Human2 (gold rows only)
        mask_gold = h2.notna()
        gold = h2[mask_gold].astype(int)
        pred = gpt_lab[mask_gold]

        n_gold = int(mask_gold.sum())
        n_pred = int(pred.notna().sum())

        # conditional on prediction
        mask_pred = pred.notna()
        cond_correct = int((pred[mask_pred].astype(int).values == gold[mask_pred].values).sum())
        cond_total = int(mask_pred.sum())
        cond_acc = cond_correct / cond_total if cond_total else float("nan")

        # strict (missing counts as wrong)
        strict_correct = int((pred.notna() & (pred.astype("Int64") == gold)).sum())
        strict_acc = strict_correct / n_gold if n_gold else float("nan")

        print("\nGPT accuracy (vs Human2 gold)")
        print("- GPT discretization: score>0 -> +1, score<0 -> -1, score==0 -> 0")
        print(f"- gold n={n_gold}")
        print(f"- GPT predicted n={n_pred} (coverage={n_pred/n_gold:.4f})")
        print(f"- accuracy (conditional on GPT prediction)={cond_acc:.4f} ({cond_correct}/{cond_total})")
        print(f"- accuracy (end-to-end; GPT missing counts as wrong)={strict_acc:.4f} ({strict_correct}/{n_gold})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
