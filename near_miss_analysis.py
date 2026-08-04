"""
near_miss_analysis.py — Quantifying near-miss vs. far-miss error for hard classes.

Answers the question in §Class Imbalance (lines 762-766 of the paper): among the FP/FN of
hard classes (hongdianshe, chihenshe, botaishe, liewenshe), what percentage are
near-misses (best semantic match in the SAME taxonomy branch, similarity > 0) compared to
far-misses (completely DIFFERENT branch, similarity = 0)?

Reuses the per_image data structure as in nwd_baseline.py
(ai_boxes / gt_boxes: dict[class] -> list[box]).

USAGE: Import and call analyze_near_miss(all_results) with all_results having the
same structure {model_name: {"per_image": [...]}} as used by exp_nwd_comparison() —
no need to retrain or run detector, only re-analyzing existing results.

NOTE: If your ai_boxes/gt_boxes use numeric class_ids instead of class names
(e.g., 0, 1, 2...), map them back to class names (botaishe, hongshe, ...) before
calling this function, as W_SEM below is indexed by class names matching
Table tab:semantic-matrix in the paper.
"""

import json
from pathlib import Path
from collections import defaultdict
CLASS_NAMES = {
    0: "botaishe",
    1: "hongshe",
    2: "pangdashe",
    3: "hongdianshe",
    4: "liewenshe",
    5: "chihenshe",
    6: "baitaishe",
    7: "huangtaishe",
}
# Wu-Palmer similarity matrix, taken directly from Table tab:semantic-matrix in the paper
W_SEM = {
    "botaishe":    {"botaishe": 1.000, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.333, "huangtaishe": 0.333},
    "hongshe":     {"botaishe": 0.000, "hongshe": 1.000, "pangdashe": 0.000, "hongdianshe": 0.500, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "pangdashe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 1.000, "hongdianshe": 0.000, "liewenshe": 0.500, "chihenshe": 0.500, "baitaishe": 0.000, "huangtaishe": 0.000},
    "hongdianshe": {"botaishe": 0.000, "hongshe": 0.500, "pangdashe": 0.000, "hongdianshe": 1.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "liewenshe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 0.500, "hongdianshe": 0.000, "liewenshe": 1.000, "chihenshe": 0.500, "baitaishe": 0.000, "huangtaishe": 0.000},
    "chihenshe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 0.500, "hongdianshe": 0.000, "liewenshe": 0.500, "chihenshe": 1.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "baitaishe":   {"botaishe": 0.333, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 1.000, "huangtaishe": 0.667},
    "huangtaishe": {"botaishe": 0.333, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.667, "huangtaishe": 1.000},
}

# 4 hardest classes according to Fig perclass-ap: mean AP@[.5:.95] < 0.13
HARD_CLASSES = {"botaishe", "hongdianshe", "liewenshe", "chihenshe"}


def _best_semantic_match(cls_k, other_side_classes):
    """Highest similarity of cls_k compared to actual labels on the opposite side
    (identical to c_k rule in Eq. csem-percls of the paper, except here we keep
    track of which label gives the best match for manual auditing if needed)."""
    if not other_side_classes:
        return 0.0, None
    best_sim, best_cls = 0.0, None
    for j in other_side_classes:
        sim = W_SEM.get(cls_k, {}).get(j, 0.0)
        if sim > best_sim:
            best_sim, best_cls = sim, j
    return best_sim, best_cls


def analyze_near_miss(all_results, hard_classes=HARD_CLASSES):

    print("\n" + "=" * 70)
    print("NEAR-MISS vs FAR-MISS ANALYSIS — Hard Classes")
    print("=" * 70)

    summary_rows = []

    # ==========================================================
    # LOOP OVER MODELS
    # ==========================================================
    for model_name, data in all_results.items():

        per_image = data["per_image"]

        # Records for current model ONLY
        records = []

        # ======================================================
        # LOOP OVER IMAGES
        # ======================================================
        for img in per_image:

            ai_boxes = img["ai_boxes"]
            gt_boxes = img["gt_boxes"]

            # --------------------------------------------------
            # Convert class IDs → class names
            # --------------------------------------------------
            ai_classes = {
                CLASS_NAMES[class_id]
                for class_id in ai_boxes.keys()
                if class_id in CLASS_NAMES
            }

            gt_classes = {
                CLASS_NAMES[class_id]
                for class_id in gt_boxes.keys()
                if class_id in CLASS_NAMES
            }

            # --------------------------------------------------
            # False Positive
            # AI predicts hard class, GT does not contain it
            # --------------------------------------------------
            for k in (ai_classes - gt_classes) & hard_classes:

                sim, match = _best_semantic_match(
                    k,
                    gt_classes
                )

                records.append({
                    "image": img.get("img_name", "?"),
                    "class": k,
                    "type": "FP",
                    "best_sim": sim,
                    "matched_with": match,
                })

            # --------------------------------------------------
            # False Negative
            # GT contains hard class, AI does not predict it
            # --------------------------------------------------
            for k in (gt_classes - ai_classes) & hard_classes:

                sim, match = _best_semantic_match(
                    k,
                    ai_classes
                )

                records.append({
                    "image": img.get("img_name", "?"),
                    "class": k,
                    "type": "FN",
                    "best_sim": sim,
                    "matched_with": match,
                })

        # ======================================================
        # AGGREGATE RESULTS FOR THIS MODEL
        # ======================================================

        n_total = len(records)

        n_near = sum(
            1
            for r in records
            if r["best_sim"] > 0
        )

        n_far = n_total - n_near

        pct_near = (
            round(100 * n_near / n_total, 1)
            if n_total
            else 0.0
        )

        # ------------------------------------------------------
        # Per-class statistics
        # ------------------------------------------------------

        per_class = defaultdict(
            lambda: {
                "total": 0,
                "near": 0
            }
        )

        for r in records:

            per_class[r["class"]]["total"] += 1

            if r["best_sim"] > 0:
                per_class[r["class"]]["near"] += 1

        per_class_pct = {

            k: {
                "total": v["total"],
                "near": v["near"],
                "pct_near": round(
                    100 * v["near"] / v["total"],
                    1
                )
                if v["total"]
                else 0.0,
            }

            for k, v in per_class.items()
        }

        # ======================================================
        # ONE SUMMARY ROW PER MODEL
        # ======================================================

        summary_rows.append({

            "model": model_name,

            "total_hardclass_errors": n_total,

            "near_miss": n_near,

            "far_miss": n_far,

            "pct_near_miss": pct_near,

            "per_class": per_class_pct,

        })

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    header = (
        f"{'Model':<10} "
        f"{'Total':<8} "
        f"{'Near-miss':<10} "
        f"{'Far-miss':<10} "
        f"{'% Near':<8}"
    )

    print(f"\n{header}")
    print("-" * len(header))

    for r in summary_rows:

        print(
            f"{r['model']:<10} "
            f"{r['total_hardclass_errors']:<8} "
            f"{r['near_miss']:<10} "
            f"{r['far_miss']:<10} "
            f"{r['pct_near_miss']:<8}"
        )

    # ==========================================================
    # POOLED RESULT ACROSS ALL MODELS
    # ==========================================================

    total_all = sum(
        r["total_hardclass_errors"]
        for r in summary_rows
    )

    near_all = sum(
        r["near_miss"]
        for r in summary_rows
    )

    pooled_pct = (
        round(100 * near_all / total_all, 1)
        if total_all
        else 0.0
    )

    print(
        f"\n  Pooled across all models: "
        f"{near_all}/{total_all} "
        f"({pooled_pct}%) "
        f"of hard-class errors are near-miss confusions."
    )

    return summary_rows, pooled_pct


def save_near_miss_report(summary_rows, pooled_pct, output_dir):
    save_path = Path(output_dir) / "near_miss_analysis.json"
    with open(save_path, "w") as f:
        json.dump({"pooled_pct_near_miss": pooled_pct, "rows": summary_rows},
                   f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {save_path}")

