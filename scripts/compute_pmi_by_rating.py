#!/usr/bin/env python3
"""
Rating-stratified PMI analysis for JMIR #92325 R2, Comment #4.

Editor's concern: Jaccard differences (Δ) between positive and negative
reviews may be confounded with marginal aspect prevalence. PMI normalizes
for base rates:

    PMI(A,B) = log2( P(A∩B) / (P(A) · P(B)) )

A positive PMI means the two aspects co-occur MORE than expected given
their individual prevalences; negative PMI means LESS than expected.

We compute PMI for all 21 aspect pairs within each rating group
(positive 4-5★, negative 1-2★), then compare ΔPMI with ΔJaccard
to check whether the differential bundling story holds after
prevalence adjustment.
"""

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).parent / "layer2_data.csv"
ASPECTS = [
    "Administrative Processes",
    "Emergency Care",
    "Facility & Environment",
    "Inpatient Care",
    "Professional Quality",
    "Service Attitude",
    "Surgical & Specialty Care",
]


def load_review_aspects(path: Path):
    """Return {review_id: (star_rating, set_of_aspects)}."""
    reviews = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(float(row["review_id"]))
            star = int(float(row["star_rating"]))
            aspect = row["aspect_category"].strip()
            if rid not in reviews:
                reviews[rid] = (star, set())
            reviews[rid][1].add(aspect)
    return reviews


def split_by_rating(reviews):
    """Split into positive (4-5★) and negative (1-2★) groups."""
    pos, neg = {}, {}
    for rid, (star, aspects) in reviews.items():
        if star >= 4:
            pos[rid] = aspects
        elif star <= 2:
            neg[rid] = aspects
    return pos, neg


def compute_prevalence(group, aspects):
    """Return {aspect: proportion} within the group."""
    n = len(group)
    if n == 0:
        return {a: 0.0 for a in aspects}
    counts = defaultdict(int)
    for aspects_set in group.values():
        for a in aspects_set:
            counts[a] += 1
    return {a: counts[a] / n for a in aspects}


def compute_cooccurrence(group, aspects):
    """Return {(a,b): proportion} for all pairs."""
    n = len(group)
    if n == 0:
        return {}
    pair_counts = defaultdict(int)
    for aspects_set in group.values():
        present = sorted(aspects_set & set(aspects))
        for a, b in combinations(present, 2):
            pair_counts[(a, b)] += 1
    return {pair: count / n for pair, count in pair_counts.items()}


def compute_jaccard(group, aspects):
    """Return {(a,b): jaccard} for all pairs."""
    aspect_sets = {a: set() for a in aspects}
    for rid, aspects_set in group.items():
        for a in aspects_set:
            if a in aspect_sets:
                aspect_sets[a].add(rid)
    result = {}
    for a, b in combinations(aspects, 2):
        inter = len(aspect_sets[a] & aspect_sets[b])
        union = len(aspect_sets[a] | aspect_sets[b])
        result[(a, b)] = inter / union if union > 0 else 0.0
    return result


def compute_pmi(group, aspects):
    """Return {(a,b): pmi} for all pairs."""
    prev = compute_prevalence(group, aspects)
    cooc = compute_cooccurrence(group, aspects)
    result = {}
    for a, b in combinations(aspects, 2):
        p_ab = cooc.get((a, b), 0.0)
        p_a = prev[a]
        p_b = prev[b]
        if p_ab > 0 and p_a > 0 and p_b > 0:
            result[(a, b)] = math.log2(p_ab / (p_a * p_b))
        else:
            result[(a, b)] = float("-inf")
    return result


def compute_expected_cooccurrence(group, aspects):
    """Return {(a,b): expected_count} under independence."""
    n = len(group)
    prev = compute_prevalence(group, aspects)
    return {
        (a, b): prev[a] * prev[b] * n
        for a, b in combinations(aspects, 2)
    }


def main():
    reviews = load_review_aspects(DATA)
    pos_group, neg_group = split_by_rating(reviews)
    print(f"Positive reviews (4-5★): {len(pos_group):,}")
    print(f"Negative reviews (1-2★): {len(neg_group):,}")

    # Prevalence per group
    pos_prev = compute_prevalence(pos_group, ASPECTS)
    neg_prev = compute_prevalence(neg_group, ASPECTS)

    print("\n=== Marginal Prevalence by Rating Group ===")
    print(f"{'Aspect':<30} {'Pos %':>8} {'Neg %':>8} {'Ratio':>8}")
    print("-" * 56)
    for a in ASPECTS:
        ratio = pos_prev[a] / neg_prev[a] if neg_prev[a] > 0 else float("inf")
        print(f"{a:<30} {pos_prev[a]*100:>7.1f}% {neg_prev[a]*100:>7.1f}% {ratio:>7.2f}")

    # Jaccard
    pos_jaccard = compute_jaccard(pos_group, ASPECTS)
    neg_jaccard = compute_jaccard(neg_group, ASPECTS)

    # PMI
    pos_pmi = compute_pmi(pos_group, ASPECTS)
    neg_pmi = compute_pmi(neg_group, ASPECTS)

    # Observed vs Expected
    pos_expected = compute_expected_cooccurrence(pos_group, ASPECTS)
    neg_expected = compute_expected_cooccurrence(neg_group, ASPECTS)
    pos_cooc_counts = defaultdict(int)
    neg_cooc_counts = defaultdict(int)
    for aspects_set in pos_group.values():
        present = sorted(aspects_set & set(ASPECTS))
        for a, b in combinations(present, 2):
            pos_cooc_counts[(a, b)] += 1
    for aspects_set in neg_group.values():
        present = sorted(aspects_set & set(ASPECTS))
        for a, b in combinations(present, 2):
            neg_cooc_counts[(a, b)] += 1

    # Full comparison table
    pairs = list(combinations(ASPECTS, 2))
    rows = []
    for pair in pairs:
        a, b = pair
        j_pos = pos_jaccard[pair]
        j_neg = neg_jaccard[pair]
        dj = j_pos - j_neg
        pmi_pos = pos_pmi[pair]
        pmi_neg = neg_pmi[pair]
        dpmi = pmi_pos - pmi_neg if pmi_pos != float("-inf") and pmi_neg != float("-inf") else float("nan")
        obs_pos = pos_cooc_counts[pair]
        exp_pos = pos_expected[pair]
        obs_neg = neg_cooc_counts[pair]
        exp_neg = neg_expected[pair]
        oe_pos = obs_pos / exp_pos if exp_pos > 0 else float("nan")
        oe_neg = obs_neg / exp_neg if exp_neg > 0 else float("nan")
        rows.append({
            "pair": f"{a} ↔ {b}",
            "j_pos": j_pos, "j_neg": j_neg, "dj": dj,
            "pmi_pos": pmi_pos, "pmi_neg": pmi_neg, "dpmi": dpmi,
            "obs_pos": obs_pos, "exp_pos": exp_pos, "oe_pos": oe_pos,
            "obs_neg": obs_neg, "exp_neg": exp_neg, "oe_neg": oe_neg,
        })

    # Sort by |ΔJaccard| descending
    rows.sort(key=lambda r: abs(r["dj"]), reverse=True)

    print("\n=== Rating-Stratified Jaccard vs PMI (sorted by |ΔJaccard|) ===")
    print(f"{'Pair':<50} {'J_pos':>6} {'J_neg':>6} {'ΔJ':>7} {'PMI_p':>7} {'PMI_n':>7} {'ΔPMI':>7} {'O/E_p':>6} {'O/E_n':>6}")
    print("-" * 110)
    for r in rows:
        pmi_p_str = f"{r['pmi_pos']:>7.3f}" if r["pmi_pos"] != float("-inf") else "  -inf"
        pmi_n_str = f"{r['pmi_neg']:>7.3f}" if r["pmi_neg"] != float("-inf") else "  -inf"
        dpmi_str = f"{r['dpmi']:>7.3f}" if not math.isnan(r["dpmi"]) else "    N/A"
        print(
            f"{r['pair']:<50} "
            f"{r['j_pos']:>6.3f} {r['j_neg']:>6.3f} {r['dj']:>+7.3f} "
            f"{pmi_p_str} {pmi_n_str} {dpmi_str} "
            f"{r['oe_pos']:>6.2f} {r['oe_neg']:>6.2f}"
        )

    # Check sign agreement between ΔJaccard and ΔPMI
    print("\n=== Sign Agreement: ΔJaccard vs ΔPMI ===")
    agree = 0
    disagree = 0
    for r in rows:
        if math.isnan(r["dpmi"]):
            continue
        if (r["dj"] > 0 and r["dpmi"] > 0) or (r["dj"] < 0 and r["dpmi"] < 0) or (r["dj"] == 0 and r["dpmi"] == 0):
            agree += 1
        else:
            disagree += 1
            print(f"  DISAGREE: {r['pair']}  ΔJ={r['dj']:+.3f}  ΔPMI={r['dpmi']:+.3f}")

    print(f"\nAgree: {agree}/{ agree + disagree}  Disagree: {disagree}/{agree + disagree}")

    # Spearman correlation between ΔJaccard and ΔPMI
    valid = [(r["dj"], r["dpmi"]) for r in rows if not math.isnan(r["dpmi"])]
    if len(valid) >= 3:
        from scipy.stats import spearmanr
        dj_vals, dpmi_vals = zip(*valid)
        rho, pval = spearmanr(dj_vals, dpmi_vals)
        print(f"Spearman ρ(ΔJaccard, ΔPMI) = {rho:.4f}, p = {pval:.4e}")

    # Write CSV
    out_path = Path(__file__).parent / "pmi_vs_jaccard_by_rating.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair", "j_pos", "j_neg", "delta_jaccard",
            "pmi_pos", "pmi_neg", "delta_pmi",
            "obs_pos", "exp_pos", "oe_ratio_pos",
            "obs_neg", "exp_neg", "oe_ratio_neg",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "pair": r["pair"],
                "j_pos": round(r["j_pos"], 4),
                "j_neg": round(r["j_neg"], 4),
                "delta_jaccard": round(r["dj"], 4),
                "pmi_pos": round(r["pmi_pos"], 4) if r["pmi_pos"] != float("-inf") else "",
                "pmi_neg": round(r["pmi_neg"], 4) if r["pmi_neg"] != float("-inf") else "",
                "delta_pmi": round(r["dpmi"], 4) if not math.isnan(r["dpmi"]) else "",
                "obs_pos": r["obs_pos"],
                "exp_pos": round(r["exp_pos"], 2),
                "oe_ratio_pos": round(r["oe_pos"], 3),
                "obs_neg": r["obs_neg"],
                "exp_neg": round(r["exp_neg"], 2),
                "oe_ratio_neg": round(r["oe_neg"], 3),
            })
    print(f"\nCSV saved: {out_path}")


if __name__ == "__main__":
    main()
