"""
alpha_beta_optimization.py — Data-driven alpha/beta weighting for CCS.

PROBLEM: CCS = alpha*C_sp + beta*C_sem with default alpha=beta=0.5, acknowledged
in CCS documentation as "not optimal". Since Part 1 DOES NOT have doctor consensus
labels (which belongs to Part 2 — acceptance study), we CANNOT optimize alpha/beta
to match "clinical acceptance degree" — doing so would mix the scopes of the two parts.

Instead, this module uses standard OBJECTIVE WEIGHTING methods from Multi-Criteria
Decision Analysis (MCDM), based solely on the VARIATION/DISCRIMINATION of C_sp and C_sem
observed directly on the data — requiring no external labels:

  1. Entropy weight method (Shannon entropy) — criteria with higher variation/discrimination
     across samples receive higher weights. A classic MCDM technique (originating from
     Shannon 1948, widely applied since the 1980s).
  2. Std-proportional weighting (~ simplified CRITIC method for 2 criteria — with 2 criteria,
     the correlation term in CRITIC cancels out, leaving only the standard deviation ratio).
  3. Ranking-stability analysis — sweeps alpha from 0 to 1, checks if the ranking of 6 models
     changes, identifies the "stable range" around 0.5 and transition points where ranking changes.

IMPORTANT NOTE TO STATE IN PAPER: these weights reflect "which criterion discriminates
better STATISTICALLY across observed samples/models", NOT "whether doctors value spatial
or semantic aspects more in clinical practice". These are two different questions; the latter
requires an acceptance study.
"""

import math
import json
from pathlib import Path

import numpy as np


# ─── Objective weighting methods ────────────────────────────────────────────

def entropy_weights(csp_values, csem_values):
    """
    Shannon entropy weighting. Returns (alpha, beta), alpha+beta=1.
    csp_values, csem_values: should be ALL per-image values, pooled from all models
    (more samples lead to more stable entropy estimates).
    """
    csp = np.asarray(csp_values, dtype=np.float64)
    csem = np.asarray(csem_values, dtype=np.float64)
    eps = 1e-12
    n = len(csp)
    if n < 2:
        return 0.5, 0.5

    def _entropy(x):
        x = np.clip(x, eps, None)
        p = x / x.sum()
        k = 1.0 / math.log(n)
        return -k * np.sum(p * np.log(p))

    e_sp, e_sem = _entropy(csp), _entropy(csem)
    d_sp, d_sem = 1 - e_sp, 1 - e_sem
    if (d_sp + d_sem) <= 0:
        return 0.5, 0.5
    alpha = d_sp / (d_sp + d_sem)
    return float(alpha), float(1 - alpha)


def std_proportional_weights(csp_values, csem_values):
    """
    Standard deviation ratio weighting (~ simplified CRITIC for 2 criteria).
    Criteria with higher variation across samples receive higher weights.
    """
    csp = np.asarray(csp_values, dtype=np.float64)
    csem = np.asarray(csem_values, dtype=np.float64)
    s_sp, s_sem = csp.std(ddof=1), csem.std(ddof=1)
    if (s_sp + s_sem) <= 0:
        return 0.5, 0.5
    alpha = s_sp / (s_sp + s_sem)
    return float(alpha), float(1 - alpha)


# ─── Ranking stability analysis ─────────────────────────────────────────────

def ranking_stability_analysis(all_results, alphas=None):
    """
    For each alpha, ranks models according to CCS = alpha*C_sp + (1-alpha)*C_sem
    (using mean C_sp/C_sem for each model). Returns:
      - rankings_by_alpha: ranking dictionary corresponding to each alpha
      - transition_points: alphas where ranking changes relative to the previous alpha
      - stable_range_around_0.5: widest alpha range containing 0.5 where ranking
        remains unchanged relative to alpha=0.5
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.001, 0.05)]

    model_means = {}
    for model_name, data in all_results.items():
        per_image = data["per_image"]
        c_sp = float(np.mean([img["ccs"]["c_sp"] for img in per_image]))
        c_sem = float(np.mean([img["ccs"]["c_sem"] for img in per_image]))
        model_means[model_name] = (c_sp, c_sem)

    rankings = {}
    for a in alphas:
        b = 1 - a
        scores = {m: a * cs + b * ce for m, (cs, ce) in model_means.items()}
        ranked = [m for m, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        rankings[a] = ranked

    transition_points = []
    prev = None
    for a in alphas:
        if prev is not None and rankings[a] != prev:
            transition_points.append(a)
        prev = rankings[a]

    closest_to_half = min(alphas, key=lambda x: abs(x - 0.5))
    base_ranking = rankings[closest_to_half]
    stable_alphas = [a for a in alphas if rankings[a] == base_ranking]
    stable_lo, stable_hi = (min(stable_alphas), max(stable_alphas)) if stable_alphas else (None, None)

    return {
        "rankings_by_alpha": {f"{a:.2f}": r for a, r in rankings.items()},
        "transition_points": transition_points,
        "stable_range_around_0.5": [stable_lo, stable_hi],
        "ranking_at_0.5": base_ranking,
    }


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_alpha_beta_datadriven(all_results, output_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT: Data-Driven Alpha/Beta Weighting (replacing default 0.5/0.5)")
    print("=" * 70)

    # pool per-image C_sp, C_sem from ALL models to estimate objective weights
    all_csp, all_csem = [], []
    for data in all_results.values():
        for img in data["per_image"]:
            all_csp.append(img["ccs"]["c_sp"])
            all_csem.append(img["ccs"]["c_sem"])

    alpha_ent, beta_ent = entropy_weights(all_csp, all_csem)
    alpha_std, beta_std = std_proportional_weights(all_csp, all_csem)

    print(f"\n  Number of per-image samples pooled from {len(all_results)} models: {len(all_csp)}")
    print(f"  Entropy weight       : alpha={alpha_ent:.4f}  beta={beta_ent:.4f}")
    print(f"  Std-proportional (~CRITIC): alpha={alpha_std:.4f}  beta={beta_std:.4f}")
    print(f"  Current default       : alpha=0.5000  beta=0.5000")

    stability = ranking_stability_analysis(all_results)
    print(f"\n  Model ranking at alpha=0.5: {stability['ranking_at_0.5']}")
    print(f"  Stable alpha range around 0.5 (ranking unchanged): "
          f"[{stability['stable_range_around_0.5'][0]}, {stability['stable_range_around_0.5'][1]}]")
    if stability["transition_points"]:
        print(f"  Alpha values where ranking changes: {stability['transition_points']}")
    else:
        print("  Ranking UNCHANGED across entire alpha range 0->1 -> conclusions on the best model")
        print("  do not depend on alpha/beta choice (very strong result if true).")

    # rank at objective alpha (entropy) to compare with alpha=0.5
    a_ent_rounded = round(alpha_ent, 2)
    closest_alpha = min(stability["rankings_by_alpha"].keys(), key=lambda k: abs(float(k) - a_ent_rounded))
    ranking_at_entropy_alpha = stability["rankings_by_alpha"][closest_alpha]
    same_as_default = ranking_at_entropy_alpha == stability["ranking_at_0.5"]
    print(f"\n  Ranking at alpha=entropy({alpha_ent:.2f}): {ranking_at_entropy_alpha}")
    print(f"  Matches ranking at alpha=0.5? {'YES' if same_as_default else 'NO'}")

    result = {
        "n_pooled_samples": len(all_csp),
        "entropy_weight": {"alpha": round(alpha_ent, 4), "beta": round(beta_ent, 4)},
        "std_proportional_weight": {"alpha": round(alpha_std, 4), "beta": round(beta_std, 4)},
        "default_weight": {"alpha": 0.5, "beta": 0.5},
        "ranking_stability": stability,
        "ranking_matches_default_at_entropy_alpha": same_as_default,
    }
    save_path = Path(output_dir) / "alpha_beta_datadriven.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return result

