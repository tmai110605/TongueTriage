"""
Additional Rebuttal Experiments:
  1. NMS IoU Threshold Sensitivity Sweep (tau_nms in [0.30, 0.80])
  2. Annotation Noise Simulation (Spatial Bounding Box Jitter & Semantic Label Corruption)
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure root dir is in path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from ccs import (
    CLASS_NAMES_8,
    build_semantic_matrix,
    compute_ccs,
    compute_ccs_instance_hungarian,
    spatial_concordance_for_pair,
)

OUTPUT_DIR = ROOT_DIR / "runs" / "experiments"
SIVP_DIR = ROOT_DIR / "CCS_SIVP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SIVP_DIR.mkdir(parents=True, exist_ok=True)


def _compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def apply_nms_to_boxes(boxes_dict, nms_thresh, conf_thresh=0.25):
    """Apply Non-Maximum Suppression within each class."""
    suppressed = {}
    for cls_id, b_list in boxes_dict.items():
        # filter by conf
        valid_b = [b for b in b_list if b[4] >= conf_thresh]
        if not valid_b:
            continue
        # sort descending by conf
        valid_b = sorted(valid_b, key=lambda x: x[4], reverse=True)
        keep = []
        while valid_b:
            curr = valid_b.pop(0)
            keep.append(curr)
            valid_b = [b for b in valid_b if _compute_iou(curr[:4], b[:4]) < nms_thresh]
        suppressed[cls_id] = keep
    return suppressed


# ─── 1. NMS Threshold Sensitivity ──────────────────────────────────────────

def run_nms_sweep(cached_data, sem_matrix):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: NMS IoU Threshold Sensitivity Sweep (0.30 to 0.80)")
    print("=" * 70)
    
    nms_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    nms_results = {}
    
    plt.figure(figsize=(7.5, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    color_map = {
        "yolov8n": "#1f77b4", "yolov8m": "#ff7f0e",
        "yolov10n": "#2ca02c", "yolov10m": "#d62728",
        "yolov11n": "#9467bd", "yolov11m": "#8c564b",
    }
    
    for model_name, img_records in cached_data.items():
        ccs_by_nms = []
        for nms_t in nms_thresholds:
            scores = []
            for rec in img_records:
                ai_boxes = apply_nms_to_boxes(rec["raw_boxes"], nms_thresh=nms_t, conf_thresh=0.25)
                res = compute_ccs(ai_boxes, rec["gt_boxes"], sem_matrix)
                scores.append(res["ccs"])
            ccs_by_nms.append(float(np.mean(scores)))
            
        nms_results[model_name] = {
            "nms_thresholds": nms_thresholds,
            "ccs_scores": ccs_by_nms,
            "min_ccs": float(np.min(ccs_by_nms)),
            "max_ccs": float(np.max(ccs_by_nms)),
            "std_ccs": float(np.std(ccs_by_nms)),
        }
        
        print(f"[{model_name:<9}] NMS Range: [{min(ccs_by_nms):.4f}, {max(ccs_by_nms):.4f}] | Std: {np.std(ccs_by_nms):.5f}")
        
        plt.plot(
            nms_thresholds, ccs_by_nms,
            label=f"{model_name} (mean={np.mean(ccs_by_nms):.4f})",
            color=color_map.get(model_name, None),
            linewidth=2.0,
            marker='s', markersize=5
        )
        
    plt.title("CCS Invariance across NMS IoU Thresholds ($\\tau_{\\mathrm{NMS}}$)", fontsize=12, pad=10, fontweight="bold")
    plt.xlabel("NMS IoU Threshold ($\\tau_{\\mathrm{NMS}}$)", fontsize=11)
    plt.ylabel("Continuous Concordance Score (CCS)", fontsize=11)
    plt.ylim(0.58, 0.68)
    plt.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default NMS Operating Point (0.50)")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="center left")
    plt.tight_layout()
    
    p1 = OUTPUT_DIR / "fig_nms_sensitivity.pdf"
    p2 = SIVP_DIR / "fig_nms_sensitivity.pdf"
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()
    print(f"[INFO] Saved NMS sensitivity plot to {p1} and {p2}")
    
    return nms_results


# ─── 2. Annotation Noise Simulation ────────────────────────────────────────

def run_annotation_noise_simulation(cached_data, sem_matrix):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Annotation Noise Simulation (Spatial Jitter & Semantic Flip)")
    print("=" * 70)
    
    np.random.seed(42)
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # We test on YOLOv8m (representative model)
    records = cached_data["yolov8m"]
    
    spatial_ccs_means = []
    spatial_f1_means = []
    semantic_ccs_means = []
    semantic_f1_means = []
    
    for noise in noise_levels:
        # A. Spatial Jitter Noise
        ccs_spat = []
        f1_spat = []
        for rec in records:
            ai_boxes = apply_nms_to_boxes(rec["raw_boxes"], nms_thresh=0.5, conf_thresh=0.25)
            # Add spatial noise to GT boxes
            noisy_gt = {}
            for cls_id, b_list in rec["gt_boxes"].items():
                noisy_list = []
                for b in b_list:
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    dx = np.random.normal(0, noise * w)
                    dy = np.random.normal(0, noise * h)
                    dw = np.random.normal(0, noise * w * 0.5)
                    dh = np.random.normal(0, noise * h * 0.5)
                    nb = (
                        max(0, b[0] + dx - dw/2),
                        max(0, b[1] + dy - dh/2),
                        b[2] + dx + dw/2,
                        b[3] + dy + dh/2,
                    )
                    noisy_list.append(nb)
                noisy_gt[cls_id] = noisy_list
                
            # Compute CCS
            res_ccs = compute_ccs(ai_boxes, noisy_gt, sem_matrix)
            ccs_spat.append(res_ccs["ccs"])
            
            # Compute IoU@0.5 F1 proxy
            tp = fp = fn = 0
            for cls_id in set(ai_boxes.keys()) | set(noisy_gt.keys()):
                ai_l = [b[:4] for b in ai_boxes.get(cls_id, [])]
                gt_l = noisy_gt.get(cls_id, [])
                matched = 0
                for a_box in ai_l:
                    if any(_compute_iou(a_box, g_box) >= 0.5 for g_box in gt_l):
                        matched += 1
                tp += matched
                fp += (len(ai_l) - matched)
                fn += max(0, len(gt_l) - matched)
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1_spat.append(f1)
            
        spatial_ccs_means.append(float(np.mean(ccs_spat)))
        spatial_f1_means.append(float(np.mean(f1_spat)))
        
        # B. Semantic Label Noise (random flip with probability `noise`)
        ccs_sem = []
        f1_sem = []
        for rec in records:
            ai_boxes = apply_nms_to_boxes(rec["raw_boxes"], nms_thresh=0.5, conf_thresh=0.25)
            noisy_gt_labels = {}
            for cls_id, b_list in rec["gt_boxes"].items():
                target_cls = cls_id
                if np.random.rand() < noise:
                    # flip to random other class
                    target_cls = int(np.random.choice([c for c in range(8) if c != cls_id]))
                if target_cls not in noisy_gt_labels:
                    noisy_gt_labels[target_cls] = []
                noisy_gt_labels[target_cls].extend(b_list)
                
            res_ccs = compute_ccs(ai_boxes, noisy_gt_labels, sem_matrix)
            ccs_sem.append(res_ccs["ccs"])
            
            # IoU F1 on corrupted labels
            tp = fp = fn = 0
            for cls_id in set(ai_boxes.keys()) | set(noisy_gt_labels.keys()):
                ai_l = [b[:4] for b in ai_boxes.get(cls_id, [])]
                gt_l = noisy_gt_labels.get(cls_id, [])
                matched = 0
                for a_box in ai_l:
                    if any(_compute_iou(a_box, g_box) >= 0.5 for g_box in gt_l):
                        matched += 1
                tp += matched
                fp += (len(ai_l) - matched)
                fn += max(0, len(gt_l) - matched)
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1_sem.append(f1)
            
        semantic_ccs_means.append(float(np.mean(ccs_sem)))
        semantic_f1_means.append(float(np.mean(f1_sem)))
        
    print("\n--- Spatial Bounding Box Jitter Results ---")
    for n_lvl, c_val, f_val in zip(noise_levels, spatial_ccs_means, spatial_f1_means):
        print(f"Noise {n_lvl*100:4.1f}% | CCS: {c_val:.4f} (Drop: {spatial_ccs_means[0]-c_val:+.4f}) | F1@0.5: {f_val:.4f} (Drop: {spatial_f1_means[0]-f_val:+.4f})")
        
    print("\n--- Semantic Label Noise Results ---")
    for n_lvl, c_val, f_val in zip(noise_levels, semantic_ccs_means, semantic_f1_means):
        print(f"Noise {n_lvl*100:4.1f}% | CCS: {c_val:.4f} (Drop: {semantic_ccs_means[0]-c_val:+.4f}) | F1@0.5: {f_val:.4f} (Drop: {semantic_f1_means[0]-f_val:+.4f})")

    # Plot 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    pct_noise = [n * 100 for n in noise_levels]
    
    # Left: Spatial Jitter
    ax1.plot(pct_noise, spatial_ccs_means, 'o-', color='#1f77b4', linewidth=2.2, label='CCS (Continuous)')
    ax1.plot(pct_noise, spatial_f1_means, 's--', color='#d62728', linewidth=2.0, label='F1@0.5 (Hard IoU)')
    ax1.set_title("(a) Spatial Annotation Jitter Noise", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Bounding Box Perturbation ($\\sigma_{\\mathrm{jitter}}$ in % of size)", fontsize=10)
    ax1.set_ylabel("Evaluation Metric Score", fontsize=10)
    ax1.legend(frameon=True, facecolor="white", fontsize=9)
    ax1.set_ylim(0.25, 0.70)
    
    # Right: Semantic Noise
    ax2.plot(pct_noise, semantic_ccs_means, 'o-', color='#1f77b4', linewidth=2.2, label='CCS (Continuous)')
    ax2.plot(pct_noise, semantic_f1_means, 's--', color='#d62728', linewidth=2.0, label='F1@0.5 (Hard IoU)')
    ax2.set_title("(b) Semantic Label Noise Corruption", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Label Corruption Probability ($p_{\\mathrm{noise}}$ in %)", fontsize=10)
    ax2.set_ylabel("Evaluation Metric Score", fontsize=10)
    ax2.legend(frameon=True, facecolor="white", fontsize=9)
    ax2.set_ylim(0.25, 0.70)
    
    plt.tight_layout()
    p1 = OUTPUT_DIR / "fig_annotation_noise.pdf"
    p2 = SIVP_DIR / "fig_annotation_noise.pdf"
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()
    print(f"[INFO] Saved Annotation Noise plot to {p1} and {p2}")
    
    return {
        "noise_levels": noise_levels,
        "spatial_noise": {"ccs": spatial_ccs_means, "f1": spatial_f1_means},
        "semantic_noise": {"ccs": semantic_ccs_means, "f1": semantic_f1_means},
    }


def main():
    cache_file = OUTPUT_DIR / "raw_predictions_cache.pkl"
    if not cache_file.exists():
        print(f"[ERROR] Prediction cache {cache_file} not found!")
        return
        
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
    print(f"[INFO] Loaded prediction cache for {len(cached_data)} models.")
    
    sem_matrix = build_semantic_matrix()
    
    res_nms = run_nms_sweep(cached_data, sem_matrix)
    res_noise = run_annotation_noise_simulation(cached_data, sem_matrix)
    
    out_payload = {
        "nms_sensitivity": res_nms,
        "annotation_noise_simulation": res_noise,
    }
    
    out_json = OUTPUT_DIR / "nms_and_noise_results.json"
    with open(out_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[SUCCESS] Saved NMS and Noise results to {out_json}")


if __name__ == "__main__":
    main()
