"""
CCS Paper Experiments v2 — Full benchmark:
  1. Threshold sensitivity (IoU / GIoU / DIoU at 0.3/0.5/0.7) vs CCS
  2. Flipped-pair detection (models whose ranking flips across thresholds)
  3. Edge-case gallery with IoU / GIoU / DIoU / CCS per-pair comparison
  4. Alpha / Beta ablation

Usage:
    python experiments_ccs_v2.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import yaml
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from map_metrics import compute_map_5095, exp_map_comparison
from nwd_baseline import exp_nwd_comparison
from stats_tests import exp_significance
from alpha_beta_optimization import exp_alpha_beta_datadriven
MAP_CONF_THRESH = 0.001  # near 0, needed to build complete Precision-Recall curves for AP
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ccs import (
    CLASS_NAMES_8,
    build_semantic_matrix,
    compute_ccs,
    parse_yolo_labels,
    parse_yolo_preds,
    spatial_concordance_for_pair,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "shezhen datasets" / "shezhenv3-8class" / "shezhenv3-8class.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "yolov8n": str(PROJECT_ROOT / "bestyolov8n.pt"),
    "yolov8m": str(PROJECT_ROOT / "bestyolov8m.pt"),
    "yolov10n": str(PROJECT_ROOT / "bestyolov10n.pt"),
    "yolov10m": str(PROJECT_ROOT / "bestyolov10m.pt"),
    "yolov11n": str(PROJECT_ROOT / "bestyolov11n.pt"),
    "yolov11m": str(PROJECT_ROOT / "bestyolov11m.pt"),
}

CHECKPOINT_IMGSZ = {k: 1024 for k in CHECKPOINTS}

CONF_THRESH = 0.25
NMS_IOU = 0.5
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_test_paths(data_yaml):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    img_dir = Path(str(cfg["path"])) / cfg["val"]
    label_dir = img_dir.parent / "labels"
    img_paths = sorted(img_dir.glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return img_paths, label_dir


def available_checkpoints():
    available = {}
    for name, path in CHECKPOINTS.items():
        if Path(path).exists():
            available[name] = path
        else:
            print(f"  [WARN] Checkpoint '{name}' not found at {path} — skipping.")
    return available


# ─── IoU / GIoU / DIoU utilities ───────────────────────────────────────────

def _compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compute_giou(box_a, box_b):
    """Generalized IoU (GIoU)."""
    iou = _compute_iou(box_a, box_b)
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    c_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = area_a + area_b - inter
    if c_area == 0:
        return iou
    return iou - (c_area - union) / c_area


def _compute_diou(box_a, box_b):
    """Distance IoU (DIoU)."""
    iou = _compute_iou(box_a, box_b)
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    d2 = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    c2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if c2 == 0:
        return iou
    return iou - d2 / c2


# ─── Box matching (threshold-based) ─────────────────────────────────────────

def _match_boxes_for_class(ai_list, gt_list, thresh, metric="iou"):
    """Greedy per-class matching using IoU / GIoU / DIoU threshold."""
    if metric == "iou":
        sim_fn = _compute_iou
    elif metric == "giou":
        sim_fn = _compute_giou
    elif metric == "diou":
        sim_fn = _compute_diou
    else:
        raise ValueError(f"Unknown metric: {metric}")

    unmatched_gt = list(range(len(gt_list)))
    tp = fp = 0
    ai_order = sorted(range(len(ai_list)), key=lambda i: ai_list[i][4], reverse=True)
    for ai_idx in ai_order:
        a_box = ai_list[ai_idx][:4]
        best_sim, best_gt_idx = -1.0, -1
        for gt_idx in unmatched_gt:
            sim = sim_fn(a_box, gt_list[gt_idx])
            if sim > best_sim:
                best_sim, best_gt_idx = sim, gt_idx
        if best_gt_idx != -1 and best_sim >= thresh:
            tp += 1
            unmatched_gt.remove(best_gt_idx)
        else:
            fp += 1
    fn = len(unmatched_gt)
    return tp, fp, fn


def _eval_one_model(checkpoints, img_paths, label_dir, sem_matrix, model_name):
    """Run inference once per model, return cached results for all experiments."""
    ckpt_path = checkpoints[model_name]
    print(f"  Running inference for {model_name} ...")
    model = YOLO(ckpt_path)
    imgsz = CHECKPOINT_IMGSZ[model_name]
    per_image = []
    for img_path in img_paths:
        # conf=MAP_CONF_THRESH retains low-confidence boxes for full mAP calculation;
        # the ai_boxes set (conf>=0.25) remains separated as before -> existing F1/CCS results UNCHANGED.
        preds = model(img_path, imgsz=imgsz, device=DEVICE, verbose=False, iou=NMS_IOU, conf=MAP_CONF_THRESH)[0]
        ai_boxes_map = parse_yolo_preds(preds, MAP_CONF_THRESH)
        ai_boxes = parse_yolo_preds(preds, CONF_THRESH)
        orig_h, orig_w = preds.orig_shape
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_labels(label_path, (orig_w, orig_h))
        ccs_result = compute_ccs(ai_boxes, gt_boxes, sem_matrix)
        per_image.append({
            "img_name": img_path.name,
            "img_path": str(img_path),
            "ai_boxes": ai_boxes,
            "ai_boxes_map": ai_boxes_map,
            "gt_boxes": gt_boxes,
            "ccs": ccs_result,
        })
    return per_image


def _accumulate_metrics(per_image, thresh, metric="iou"):
    """Accumulate TP/FP/FN across all images using given metric+threshold."""
    metrics = {
        "tp": 0, "fp": 0, "fn": 0,
        "tp_per_class": defaultdict(int),
        "fp_per_class": defaultdict(int),
        "fn_per_class": defaultdict(int),
        "total_gt_per_class": defaultdict(int),
    }
    for img in per_image:
        ai_boxes = img["ai_boxes"]
        gt_boxes = img["gt_boxes"]
        classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
        for cls_id in classes:
            ai_list = ai_boxes.get(cls_id, [])
            gt_list = gt_boxes.get(cls_id, [])
            tp, fp, fn = _match_boxes_for_class(ai_list, gt_list, thresh, metric)
            metrics["tp"] += tp
            metrics["fp"] += fp
            metrics["fn"] += fn
            metrics["tp_per_class"][cls_id] += tp
            metrics["fp_per_class"][cls_id] += fp
            metrics["fn_per_class"][cls_id] += fn
            metrics["total_gt_per_class"][cls_id] += len(gt_list)
    return metrics


def compute_scores(metrics):
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class = {}
    all_cls = set(metrics["tp_per_class"]) | set(metrics["fp_per_class"]) | set(metrics["fn_per_class"])
    for cls_id in all_cls:
        tp_c = metrics["tp_per_class"][cls_id]
        fp_c = metrics["fp_per_class"][cls_id]
        gt_c = metrics["total_gt_per_class"][cls_id]
        p_cls = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_cls = tp_c / gt_c if gt_c > 0 else 0.0
        f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0
        per_class[int(cls_id)] = {"precision": round(p_cls, 4), "recall": round(r_cls, 4), "f1": round(f1_cls, 4)}
    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


# ─── Best-pair helper (mirrors ccs.py) ──────────────────────────────────────

def _best_matching_pair(ai_list, gt_list):
    if not ai_list or not gt_list:
        return None
    best_s, best_pair = -1.0, None
    for a_box in ai_list:
        a_xy = a_box[:4]
        for g_box in gt_list:
            s = spatial_concordance_for_pair(a_xy, g_box)
            if s > best_s:
                best_s, best_pair = s, (a_xy, g_box, s)
    return best_pair


# ─── Experiment 1: Threshold Sensitivity ────────────────────────────────────

def exp_threshold_sensitivity(all_results, sem_matrix):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Threshold Sensitivity (IoU / GIoU / DIoU vs CCS)")
    print("=" * 70)

    thresholds = [0.3, 0.5, 0.7]
    metrics_list = ["iou", "giou", "diou"]
    rows = []

    for model_name in all_results:
        model_data = all_results[model_name]
        per_image = model_data["per_image"]

        # CCS mean (threshold-independent)
        ccs_vals = [img["ccs"]["ccs"] for img in per_image]
        csp_vals = [img["ccs"]["c_sp"] for img in per_image]
        csem_vals = [img["ccs"]["c_sem"] for img in per_image]
        m_ccs = float(np.mean(ccs_vals))
        m_csp = float(np.mean(csp_vals))
        m_csem = float(np.mean(csem_vals))

        for thresh in thresholds:
            for metric in metrics_list:
                m = _accumulate_metrics(per_image, thresh, metric)
                s = compute_scores(m)
                rows.append({
                    "model": model_name,
                    "metric": metric.upper(),
                    "threshold": thresh,
                    "precision": s["precision"],
                    "recall": s["recall"],
                    "f1": s["f1"],
                    "macro_f1": s["macro_f1"],
                    "ccs": round(m_ccs, 4),
                    "c_sp": round(m_csp, 4),
                    "c_sem": round(m_csem, 4),
                })

    # Print table
    header = f"{'Model':<10} {'Metric':<6} {'Thr':<5} {'Prec':<8} {'Rec':<8} {'F1':<8} {'mF1':<8} {'CCS':<8} {'C_sp':<8} {'C_sem':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<10} {r['metric']:<6} {r['threshold']:<5.1f} {r['precision']:<8.4f} {r['recall']:<8.4f} {r['f1']:<8.4f} {r['macro_f1']:<8.4f} {r['ccs']:<8.4f} {r['c_sp']:<8.4f} {r['c_sem']:<8.4f}")

    save_path = OUTPUT_DIR / "threshold_sensitivity.json"
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return rows


# ─── Experiment 2: Flipped-pair detection ───────────────────────────────────

def exp_flipped_pairs(all_results, sem_matrix):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Flipped-Pair Detection")
    print("=" * 70)

    models = list(all_results.keys())
    thresholds = [0.3, 0.5, 0.7]
    metrics_list = ["iou", "giou", "diou"]
    flips = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mA, mB = models[i], models[j]
            f1_at_thresh = {}
            for thresh in thresholds:
                for metric in metrics_list:
                    mA_scores = compute_scores(_accumulate_metrics(all_results[mA]["per_image"], thresh, metric))
                    mB_scores = compute_scores(_accumulate_metrics(all_results[mB]["per_image"], thresh, metric))
                    winner_mF1 = "A" if mA_scores["macro_f1"] > mB_scores["macro_f1"] else "B"
                    winner_F1 = "A" if mA_scores["f1"] > mB_scores["f1"] else "B"
                    key = f"{metric.upper()}@{thresh}"
                    f1_at_thresh[key] = {"winner_f1": winner_F1, "winner_mf1": winner_mF1, "delta_f1": round(mA_scores["f1"] - mB_scores["f1"], 4)}

            # CCS ranking
            ccs_A = np.mean([img["ccs"]["ccs"] for img in all_results[mA]["per_image"]])
            ccs_B = np.mean([img["ccs"]["ccs"] for img in all_results[mB]["per_image"]])
            ccs_winner = "A" if ccs_A > ccs_B else "B"

            # Check if any threshold-based ranking disagrees with CCS
            disagreements = []
            for key, val in f1_at_thresh.items():
                if val["winner_f1"] != ccs_winner:
                    disagreements.append(key)
            if disagreements:
                flips.append({
                    "model_A": mA, "model_B": mB,
                    "ccs_A": round(float(ccs_A), 4),
                    "ccs_B": round(float(ccs_B), 4),
                    "ccs_winner": ccs_winner,
                    "f1_details": f1_at_thresh,
                    "disagreements": disagreements,
                })

    print(f"\n  Flipped pairs found: {len(flips)}")
    for f in flips:
        print(f"  [{f['model_A']} vs {f['model_B']}] CCS: {f['ccs_winner']} wins (A={f['ccs_A']}, B={f['ccs_B']})")
        print(f"    Disagreements at: {', '.join(f['disagreements'])}")
        for k, v in f['f1_details'].items():
            print(f"      {k}: F1 winner={v['winner_f1']}, delta={v['delta_f1']}")

    save_path = OUTPUT_DIR / "flipped_pairs.json"
    with open(save_path, "w") as f:
        json.dump(flips, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return flips


# ─── Experiment 3: Edge-case gallery with 4-metric comparison ───────────────

def _class_name(cid):
    return CLASS_NAMES_8[cid] if cid < len(CLASS_NAMES_8) else str(cid)

def exp_edge_cases(all_results, sem_matrix, max_per_reason=3):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Edge-Case Gallery (IoU / GIoU / DIoU / Gaussian)")
    print("=" * 70)

    cases = []
    reason_counts = {}

    for model_name, model_data in all_results.items():
        print(f"  Scanning {model_name} ...")
        rc = {"iou_ccs_gap": 0, "sem_mismatch": 0, "multi_box": 0}
        for img in model_data["per_image"]:
            if all(c >= max_per_reason for c in rc.values()):
                break
            ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
            ccs_r = img["ccs"]
            ai_labels = set(ai_boxes.keys())
            gt_labels = set(gt_boxes.keys())
            common = ai_labels & gt_labels

            low_iou_high_ccs = False
            pair_details = []
            for k in common:
                pair = _best_matching_pair(ai_boxes[k], gt_boxes[k])
                if pair is None:
                    continue
                a_xy, g_xy, s_k = pair
                iou_v = _compute_iou(a_xy, g_xy)
                giou_v = _compute_giou(a_xy, g_xy)
                diou_v = _compute_diou(a_xy, g_xy)
                pair_details.append({
                    "class": int(k), "class_name": _class_name(k),
                    "iou": round(iou_v, 4), "giou": round(giou_v, 4),
                    "diou": round(diou_v, 4), "gaussian_sk": round(s_k, 4),
                })
                if iou_v < 0.45 and s_k > 0.6:
                    low_iou_high_ccs = True

            sem_mismatch = False
            for k in (ai_labels - gt_labels):
                if gt_labels and max(sem_matrix[k, j] for j in gt_labels) > 0.4:
                    sem_mismatch = True
                    break
            if not sem_mismatch:
                for k in (gt_labels - ai_labels):
                    if ai_labels and max(sem_matrix[k, j] for j in ai_labels) > 0.4:
                        sem_mismatch = True
                        break

            gt_multi = any(len(v) > 1 for v in gt_boxes.values())
            ai_multi = any(len(v) > 1 for v in ai_boxes.values())
            multi_box = gt_multi or ai_multi

            reasons = []
            if low_iou_high_ccs and rc["iou_ccs_gap"] < max_per_reason:
                reasons.append("iou_ccs_gap")
            if sem_mismatch and rc["sem_mismatch"] < max_per_reason:
                reasons.append("sem_mismatch")
            if multi_box and rc["multi_box"] < max_per_reason:
                reasons.append("multi_box")

            if reasons:
                for r in reasons:
                    rc[r] += 1
                cases.append({
                    "model": model_name,
                    "image": img["img_name"],
                    "ccs": round(ccs_r["ccs"], 4),
                    "c_sp": round(ccs_r["c_sp"], 4),
                    "c_sem": round(ccs_r["c_sem"], 4),
                    "pair_details": pair_details,
                    "ai_labels": sorted(ai_labels),
                    "gt_labels": sorted(gt_labels),
                    "ai_box_counts": {str(k): len(v) for k, v in ai_boxes.items()},
                    "gt_box_counts": {str(k): len(v) for k, v in gt_boxes.items()},
                    "reason": ";".join(reasons),
                })

    print(f"\n  Total edge cases: {len(cases)}")
    for c in cases:
        print(f"    [{c['model']}] {c['image']}: CCS={c['ccs']}")
        print(f"      Labels: GT={c['gt_labels']}, AI={c['ai_labels']}")
        for p in c["pair_details"]:
            print(f"      {p['class_name']}({p['class']}): IoU={p['iou']:.4f} GIoU={p['giou']:.4f} DIoU={p['diou']:.4f} Gaussian(s_k)={p['gaussian_sk']:.4f}")
        print(f"      Reason: {c['reason']}")

    save_path = OUTPUT_DIR / "edge_cases.json"
    with open(save_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return cases


# ─── Experiment 4: Ablation α, β ────────────────────────────────────────────

def exp_ablation(all_results, sem_matrix):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Alpha / Beta Ablation")
    print("=" * 70)

    alphas = [round(a, 1) for a in np.arange(0, 1.05, 0.1)]
    rows = []

    for model_name in all_results:
        per_image = all_results[model_name]["per_image"]
        c_sp_mean = float(np.mean([img["ccs"]["c_sp"] for img in per_image]))
        c_sem_mean = float(np.mean([img["ccs"]["c_sem"] for img in per_image]))
        for a in alphas:
            b = round(1.0 - a, 1)
            ccs = a * c_sp_mean + b * c_sem_mean
            rows.append({
                "model": model_name,
                "alpha": a,
                "beta": b,
                "ccs": round(ccs, 4),
                "c_sp": round(c_sp_mean, 4),
                "c_sem": round(c_sem_mean, 4),
            })

    print(f"\n{'Model':<10} {'α':<8} {'β':<8} {'CCS':<10} {'C_sp':<10} {'C_sem':<10}")
    print("-" * 56)
    for r in rows:
        print(f"{r['model']:<10} {r['alpha']:<8.1f} {r['beta']:<8.1f} {r['ccs']:<10.4f} {r['c_sp']:<10.4f} {r['c_sem']:<10.4f}")

    save_path = OUTPUT_DIR / "ablation_alpha_beta.json"
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return rows


# ─── Summary ranking table ──────────────────────────────────────────────────

def print_summary_ranking(all_results, map_rows=None):
    print("\n" + "=" * 70)
    print("SUMMARY: Model Ranking Comparison")
    print("=" * 70)

    map_lookup = {}
    if map_rows:
        for r in map_rows:
            if r["metric"] == "IOU":
                map_lookup[r["model"]] = r["map_50_95"]

    rows = []
    for model_name in all_results:
        per_image = all_results[model_name]["per_image"]
        ccs = float(np.mean([img["ccs"]["ccs"] for img in per_image]))
        c_sp = float(np.mean([img["ccs"]["c_sp"] for img in per_image]))
        c_sem = float(np.mean([img["ccs"]["c_sem"] for img in per_image]))
        m05 = compute_scores(_accumulate_metrics(per_image, 0.5, "iou"))
        m07 = compute_scores(_accumulate_metrics(per_image, 0.7, "iou"))
        rows.append({
            "model": model_name,
            "mAP@0.5 F1": m05["f1"],
            "mAP@0.7 F1": m07["f1"],
            "delta(F1_05→07)": round(m07["f1"] - m05["f1"], 4),
            "mAP@[.5:.95]": map_lookup.get(model_name),
            "CCS": round(ccs, 4),
            "C_sp": round(c_sp, 4),
            "C_sem": round(c_sem, 4),
        })

    rows.sort(key=lambda r: r["CCS"], reverse=True)
    header = f"{'Model':<10} {'F1@0.5':<10} {'F1@0.7':<10} {'ΔF1':<10} {'mAP[.5:.95]':<12} {'CCS':<10} {'C_sp':<10} {'C_sem':<10}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        m5095 = r["mAP@[.5:.95]"]
        m5095_str = f"{m5095:.4f}" if m5095 is not None else "N/A"
        print(f"{r['model']:<10} {r['mAP@0.5 F1']:<10.4f} {r['mAP@0.7 F1']:<10.4f} "
              f"{r['delta(F1_05→07)']:<10.4f} {m5095_str:<12} {r['CCS']:<10.4f} {r['C_sp']:<10.4f} {r['C_sem']:<10.4f}")

    save_path = OUTPUT_DIR / "summary_ranking.json"
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def print_class_mapping():
    print("\nClass mapping:")
    for i, name in enumerate(CLASS_NAMES_8):
        print(f"  {i}: {name}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CCS Paper Experiments v2 — Full Benchmark")
    print("=" * 70)

    sem_matrix = build_semantic_matrix()
    print_class_mapping()

    checkpoints = available_checkpoints()
    if not checkpoints:
        print("\n  [ERROR] No checkpoints found.")
        return

    img_paths, label_dir = load_test_paths(DATA_YAML)
    print(f"\nTest set: {len(img_paths)} images")
    print(f"Models: {list(checkpoints.keys())}")

    # Run inference once per model, cache results
    all_results = {}
    for model_name in checkpoints:
        per_image = _eval_one_model(checkpoints, img_paths, label_dir, sem_matrix, model_name)
        all_results[model_name] = {
            "per_image": per_image,
        }

    # Run all experiments
    exp_threshold_sensitivity(all_results, sem_matrix)
    exp_flipped_pairs(all_results, sem_matrix)
    exp_edge_cases(all_results, sem_matrix)
    exp_ablation(all_results, sem_matrix)

    metric_fns = {"iou": _compute_iou, "giou": _compute_giou, "diou": _compute_diou}
    map_rows = exp_map_comparison(all_results, metric_fns, OUTPUT_DIR)
    nwd_rows = exp_nwd_comparison(all_results, OUTPUT_DIR)
    sig_results = exp_significance(all_results, _match_boxes_for_class, output_dir=OUTPUT_DIR)
    ab_results = exp_alpha_beta_datadriven(all_results, OUTPUT_DIR)

    print_summary_ranking(all_results, map_rows)
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
