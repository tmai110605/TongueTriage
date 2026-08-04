"""
stats_tests.py — Statistical significance testing for CCS Paper Experiments.

Addresses key gap: all model comparisons in v2 (flipped_pairs, summary_ranking)
were previously based solely on MEAN scores — with no confidence intervals or hypothesis
tests. With 6 models differing by a few thousandths in CCS (e.g., yolov10m 0.6413 vs
yolov8m 0.6421), it is impossible to determine whether differences are meaningful or noise.

This module reuses test set images as resampling units (each image treated independently),
requiring no retraining or additional doctor labels:

  1. bootstrap_score / paired_bootstrap_diff — image-level bootstrap providing 95% CIs
     and two-sided p-values for DIFFERENCES in any score function (CCS, F1, mAP...).
  2. wilcoxon_signed_rank — paired non-parametric test on per-image score arrays
     (e.g., per-image CCS of Model A vs. Model B) — independent of scipy.
  3. spearman_rank_correlation — rank correlation between 2 model ranking methods
     (e.g., ranking by CCS vs. ranking by F1@0.5) — independent of scipy.

MULTIPLE COMPARISONS WARNING: with 6 models yielding C(6,2)=15 pairs, testing 15 times
at alpha=0.05 significantly inflates false positive risk. exp_significance() reports both
raw p-values AND Bonferroni-adjusted threshold (alpha/15) — adjusted thresholds should be
used for paper conclusions.
"""

import math
import json
from pathlib import Path

import numpy as np


# ─── Bootstrap (image-level resampling) ─────────────────────────────────────

def bootstrap_score(per_image, score_fn, n_boot=2000, ci=0.95, seed=42):
    """
    95% CI for any score function score_fn(list_per_image_dict) -> float,
    via image resampling with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(per_image)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled = [per_image[i] for i in idx]
        boot_vals[b] = score_fn(resampled)
    point = score_fn(per_image)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_vals, [alpha, 1 - alpha])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_diff(per_image_a, per_image_b, score_fn, n_boot=5000, seed=42):
    """
    per_image_a, per_image_b: SAME image list, SAME order (2 models evaluated on same test set) ->
    each resample step draws the SAME image indices for both models (preserving paired structure),
    then computes score_fn(A) - score_fn(B).

    score_fn: callable(list_per_image_dict) -> float. Can be mean CCS (fast) or pooled F1
    (more expensive due to box matching) — for expensive score_fn, reduce n_boot to ~500-1000.
    """
    assert len(per_image_a) == len(per_image_b), "Both models must run on the exact same image list and order"
    n = len(per_image_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled_a = [per_image_a[i] for i in idx]
        resampled_b = [per_image_b[i] for i in idx]
        diffs[b] = score_fn(resampled_a) - score_fn(resampled_b)
    observed = score_fn(per_image_a) - score_fn(per_image_b)
    if observed >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    p = float(min(1.0, p))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "observed_diff": float(observed),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": p,
        "significant_0.05": bool(p < 0.05),
    }


# ─── Wilcoxon signed-rank (paired per-image test, custom implementation) ───

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilcoxon_signed_rank(values_a, values_b):
    """
    Wilcoxon signed-rank test (normal approximation, no scipy dependency) for 2
    EQUAL LENGTH score arrays, paired by index (e.g. per-image CCS for Model A/B).
    Requires n >= ~20 for reliable normal approximation (test set detection easily satisfies this).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return {"n_effective": int(n), "note": "n too small (<10) for reliable normal approximation"}

    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    signed_ranks = ranks * np.sign(d)
    W_pos = signed_ranks[signed_ranks > 0].sum()
    W_neg = -signed_ranks[signed_ranks < 0].sum()
    W = min(W_pos, W_neg)

    mean_W = n * (n + 1) / 4.0
    std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mean_W) / std_W if std_W > 0 else 0.0
    p_value = min(1.0, 2 * (1 - _norm_cdf(abs(z))))

    return {
        "n_effective": int(n),
        "W_statistic": float(W),
        "z": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05),
    }


def per_image_f1_array(per_image, thresh, metric, match_boxes_for_class_fn):
    """
    Computes F1 INDIVIDUALLY per image (without pooling first), producing a paired
    array for Wilcoxon test. match_boxes_for_class_fn: pass _match_boxes_for_class
    from experiments_ccs_v2.py.
    NOTE: Per-image F1 is prone to 0/1 extremes (images with few boxes) -> noisier than
    per-image CCS; Wilcoxon on per-image F1 should be viewed as supplementary, not
    replacing pooled F1 paired bootstrap (paired_bootstrap_diff).
    """
    f1s = []
    for img in per_image:
        ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
        classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
        tp = fp = fn = 0
        for cls_id in classes:
            t, f, n = match_boxes_for_class_fn(ai_boxes.get(cls_id, []), gt_boxes.get(cls_id, []), thresh, metric)
            tp += t
            fp += f
            fn += n
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return np.array(f1s)


# ─── Spearman rank correlation (custom implementation) ───────────────────────

def _rank_with_ties(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(values_a, values_b):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    ra, rb = _rank_with_ties(a), _rank_with_ties(b)
    rac, rbc = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt((rac ** 2).sum() * (rbc ** 2).sum())
    return float((rac * rbc).sum() / denom) if denom > 0 else 0.0


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_significance(all_results, match_boxes_for_class_fn, n_boot_ccs=2000, n_boot_f1=800, output_dir="."):
    """
    Runs full significance testing pipeline for all model pairs:
      - paired bootstrap + Wilcoxon on mean / per-image CCS
      - paired bootstrap on pooled F1@0.5 (IoU)
      - Spearman correlation between CCS ranking and F1@0.5 ranking across all models
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Statistical Significance Testing")
    print("=" * 70)

    models = list(all_results.keys())
    n_pairs = len(models) * (len(models) - 1) // 2
    alpha_bonferroni = 0.05 / n_pairs if n_pairs > 0 else 0.05
    print(f"\n  Number of model pairs: {n_pairs} -> Bonferroni-adjusted alpha threshold = {alpha_bonferroni:.5f}")

    def ccs_mean_fn(imgs):
        return float(np.mean([img["ccs"]["ccs"] for img in imgs]))

    # Pooled F1@0.5 (IoU) — requires _match_boxes_for_class_fn from main script
    def make_f1_fn(thresh, metric):
        def f1_fn(imgs):
            tp = fp = fn = 0
            for img in imgs:
                ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
                classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
                for cls_id in classes:
                    t, f, n = match_boxes_for_class_fn(ai_boxes.get(cls_id, []), gt_boxes.get(cls_id, []), thresh, metric)
                    tp += t; fp += f; fn += n
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1_fn

    f1_05_fn = make_f1_fn(0.5, "iou")

    pairwise_results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mA, mB = models[i], models[j]
            imgs_a = all_results[mA]["per_image"]
            imgs_b = all_results[mB]["per_image"]

            ccs_a = [img["ccs"]["ccs"] for img in imgs_a]
            ccs_b = [img["ccs"]["ccs"] for img in imgs_b]

            boot_ccs = paired_bootstrap_diff(imgs_a, imgs_b, ccs_mean_fn, n_boot=n_boot_ccs)
            wilcox_ccs = wilcoxon_signed_rank(ccs_a, ccs_b)
            boot_f1 = paired_bootstrap_diff(imgs_a, imgs_b, f1_05_fn, n_boot=n_boot_f1)

            result = {
                "model_A": mA,
                "model_B": mB,
                "ccs_bootstrap": boot_ccs,
                "ccs_wilcoxon": wilcox_ccs,
                "f1_0.5_bootstrap": boot_f1,
                "significant_ccs_bonferroni": bool(boot_ccs["p_value"] < alpha_bonferroni),
                "significant_f1_bonferroni": bool(boot_f1["p_value"] < alpha_bonferroni),
            }
            pairwise_results.append(result)

            print(f"\n  [{mA} vs {mB}]")
            print(f"    CCS: diff={boot_ccs['observed_diff']:+.4f}  95%CI=[{boot_ccs['ci95_low']:+.4f}, {boot_ccs['ci95_high']:+.4f}]  "
                  f"p={boot_ccs['p_value']:.4f}  Wilcoxon p={wilcox_ccs.get('p_value', 'N/A')}")
            print(f"    F1@0.5(IoU): diff={boot_f1['observed_diff']:+.4f}  95%CI=[{boot_f1['ci95_low']:+.4f}, {boot_f1['ci95_high']:+.4f}]  "
                  f"p={boot_f1['p_value']:.4f}")
            if boot_ccs["p_value"] >= alpha_bonferroni:
                print(f"    -> NOT statistically significant after Bonferroni correction (p >= {alpha_bonferroni:.5f})")

    # Spearman correlation between CCS ranking and F1@0.5 ranking across all models
    ccs_means = [ccs_mean_fn(all_results[m]["per_image"]) for m in models]
    f1_means = [f1_05_fn(all_results[m]["per_image"]) for m in models]
    rho = spearman_rank_correlation(ccs_means, f1_means)

    print(f"\n  Spearman rho(CCS ranking, F1@0.5 IoU ranking) across {len(models)} models = {rho:.4f}")
    print("  (rho close to 1 means the two ranking methods are virtually identical overall — if CCS")
    print("   substantially alters model rankings compared to F1, rho will be noticeably below 1.)")

    output = {
        "n_models": len(models),
        "n_pairs": n_pairs,
        "bonferroni_alpha": round(alpha_bonferroni, 6),
        "spearman_ccs_vs_f1_0.5": round(rho, 4),
        "pairwise": pairwise_results,
    }
    save_path = Path(output_dir) / "significance_tests.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return output

