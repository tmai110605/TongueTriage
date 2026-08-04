"""
map_metrics.py — COCO-style mAP@[0.5:0.95] baseline for CCS Paper Experiments v2.

This is a critical baseline because mAP@[0.5:0.95] (mean AP across 10 IoU thresholds:
0.50, 0.55, ..., 0.95) is the detection community's standard solution for the exact
"threshold sensitivity" problem that CCS addresses. If the paper does not compare
against mAP@[0.5:0.95], reviewers will ask "why not use COCO mAP directly?" — this
module answers that question using empirical data.

DO NOT modify ccs.py. Simply import the 3 functions `_compute_iou`, `_compute_giou`, `_compute_diou`
from `experiments_ccs_v2.py` (do not rewrite) and reuse the `per_image` structure created by `_eval_one_model()`.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ─── AP core (COCO 101-point interpolation) ─────────────────────────────────

def _compute_ap_101point(recalls, precisions):
    """101-point interpolated AP, following standard COCO eval (pycocotools)."""
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)

    # force precision to be monotonically decreasing from right to left (standard PR curve envelope)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for rt in recall_thresholds:
        idxs = np.where(recalls >= rt)[0]
        p = precisions[idxs].max() if len(idxs) > 0 else 0.0
        ap += p
    return ap / 101.0


def _ap_for_class_at_iou(dets, gts_by_img, iou_thresh, sim_fn):
    """
    AP for ONE class at ONE specific IoU/GIoU/DIoU threshold, aggregated across all images.

    dets: list of tuples (img_idx, box[x1,y1,x2,y2], score) — MUST be the complete set,
          NOT filtered by confidence beforehand, otherwise the PR curve will be truncated
          and calculated AP will be lower than actual.
    gts_by_img: dict img_idx -> list of GT boxes [x1,y1,x2,y2] for that class.
    sim_fn: one of _compute_iou / _compute_giou / _compute_diou.
    """
    total_gt = sum(len(v) for v in gts_by_img.values())
    if total_gt == 0:
        return None  # class does not appear in GT -> AP undefined (COCO convention: skip)

    dets_sorted = sorted(dets, key=lambda d: d[2], reverse=True)
    matched = {img_idx: set() for img_idx in gts_by_img}

    n = len(dets_sorted)
    tps = np.zeros(n)
    fps = np.zeros(n)

    for i, (img_idx, box, _score) in enumerate(dets_sorted):
        gt_list = gts_by_img.get(img_idx, [])
        best_sim, best_j = -1.0, -1
        for j, gbox in enumerate(gt_list):
            if j in matched[img_idx]:
                continue
            sim = sim_fn(box, gbox)
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_j != -1 and best_sim >= iou_thresh:
            tps[i] = 1.0
            matched[img_idx].add(best_j)
        else:
            fps[i] = 1.0

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _compute_ap_101point(recalls, precisions)


# ─── Per-model mAP@[0.5:0.95] ────────────────────────────────────────────────

def compute_map_5095(per_image, metric_fns, metric="iou", box_field="ai_boxes_map"):
    """
    Computes mAP@[.5:.95], mAP@.5, mAP@.75 for ONE model, using IoU/GIoU/DIoU metric.

    per_image: list of dicts returned by _eval_one_model() (cached).
    metric_fns: dict {"iou": _compute_iou, "giou": _compute_giou, "diou": _compute_diou}
                (passed from existing functions in experiments_ccs_v2.py, not rewritten).
    box_field: key containing UNFILTERED confidence detections — default "ai_boxes_map".
    """
    sim_fn = metric_fns[metric]
    iou_range = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]  # 0.50..0.95

    all_classes = set()
    for img in per_image:
        all_classes |= set(img[box_field].keys())
        all_classes |= set(img["gt_boxes"].keys())

    class_dets = defaultdict(list)
    class_gts = defaultdict(dict)
    for img_idx, img in enumerate(per_image):
        for cls_id, boxes in img[box_field].items():
            for b in boxes:
                class_dets[cls_id].append((img_idx, b[:4], b[4]))
        for cls_id, boxes in img["gt_boxes"].items():
            class_gts[cls_id][img_idx] = boxes

    per_class = {}
    for cls_id in sorted(all_classes):
        dets = class_dets.get(cls_id, [])
        gts_by_img = class_gts.get(cls_id, {})
        if sum(len(v) for v in gts_by_img.values()) == 0:
            continue
        ap_per_iou = {}
        for iou_t in iou_range:
            ap = _ap_for_class_at_iou(dets, gts_by_img, iou_t, sim_fn)
            ap_per_iou[f"{iou_t:.2f}"] = round(ap, 4) if ap is not None else 0.0
        ap_5095 = float(np.mean(list(ap_per_iou.values())))
        per_class[int(cls_id)] = {
            "ap_per_iou": ap_per_iou,
            "ap_50_95": round(ap_5095, 4),
            "ap_50": ap_per_iou.get("0.50", 0.0),
            "ap_75": ap_per_iou.get("0.75", 0.0),
        }

    if not per_class:
        return {"map_50_95": 0.0, "map_50": 0.0, "map_75": 0.0, "per_class": {}}

    return {
        "map_50_95": round(float(np.mean([v["ap_50_95"] for v in per_class.values()])), 4),
        "map_50": round(float(np.mean([v["ap_50"] for v in per_class.values()])), 4),
        "map_75": round(float(np.mean([v["ap_75"] for v in per_class.values()])), 4),
        "per_class": per_class,
    }


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_map_comparison(all_results, metric_fns, output_dir):
    """
    Runs compute_map_5095 for all models x {iou, giou, diou}, prints table,
    saves map_5095.json, and returns rows for main() to merge into summary_ranking.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: COCO-style mAP@[0.5:0.95] (IoU / GIoU / DIoU) vs CCS")
    print("=" * 70)

    rows = []
    per_class_rows = []  # retain per-class breakdown for all 8 classes
    for model_name, model_data in all_results.items():
        per_image = model_data["per_image"]
        ccs_mean = float(np.mean([img["ccs"]["ccs"] for img in per_image]))
        for metric in ("iou", "giou", "diou"):
            m = compute_map_5095(per_image, metric_fns, metric=metric)
            rows.append({
                "model": model_name,
                "metric": metric.upper(),
                "map_50_95": m["map_50_95"],
                "map_50": m["map_50"],
                "map_75": m["map_75"],
                "ccs": round(ccs_mean, 4),
            })
            for cls_id, cls_stats in m["per_class"].items():
                per_class_rows.append({
                    "model": model_name,
                    "metric": metric.upper(),
                    "class_id": cls_id,
                    "ap_50_95": cls_stats["ap_50_95"],
                    "ap_50": cls_stats["ap_50"],
                    "ap_75": cls_stats["ap_75"],
                })

    header = f"{'Model':<10} {'Metric':<6} {'mAP@.5:.95':<12} {'mAP@.5':<10} {'mAP@.75':<10} {'CCS':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<10} {r['metric']:<6} {r['map_50_95']:<12.4f} "
              f"{r['map_50']:<10.4f} {r['map_75']:<10.4f} {r['ccs']:<8.4f}")

    save_path = Path(output_dir) / "map_5095.json"
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved to {save_path}")

    per_class_path = Path(output_dir) / "map_5095_per_class.json"
    with open(per_class_path, "w") as f:
        json.dump(per_class_rows, f, indent=2)
    print(f"  Saved per-class breakdown to {per_class_path}")

    return rows