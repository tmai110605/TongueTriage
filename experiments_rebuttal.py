"""
Consolidated Rebuttal Experiments for CCS Manuscript (Springer SIVP)

Experiments:
  1. Instance-Level Hungarian Matching vs Class-Level CCS across 6 YOLO models
  2. Confidence Threshold Sweep (0.10 to 0.90) and Area Under CCS Curve (mCCS)
  3. Taxonomy Sensitivity Analysis (Default vs 3 Alternative Taxonomies)
  4. Computational Runtime Benchmark (IoU vs NWD vs Class-CCS vs Instance-CCS)
  5. Statistical Effect Sizes (Cohen's d and Hedges' g for 15 model pairs)
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats
from ultralytics import YOLO

# Ensure repository root is in path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from ccs import (
    CLASS_NAMES_8,
    TAXONOMY_TREE,
    TAXONOMY_TREE_ALT1,
    TAXONOMY_TREE_ALT2,
    TAXONOMY_TREE_ALT3,
    build_semantic_matrix,
    build_semantic_matrix_from_tree,
    compute_ccs,
    compute_ccs_instance_hungarian,
    parse_yolo_labels,
    parse_yolo_preds,
    spatial_concordance_for_pair,
)

# ─── Configuration ──────────────────────────────────────────────────────────
DATA_YAML = ROOT_DIR / "shezhen datasets" / "shezhenv3-8class" / "shezhenv3-8class.yaml"
OUTPUT_DIR = ROOT_DIR / "runs" / "experiments"
SIVP_DIR = ROOT_DIR / "CCS_SIVP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SIVP_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "yolov8n": str(ROOT_DIR / "bestyolov8n.pt"),
    "yolov8m": str(ROOT_DIR / "bestyolov8m.pt"),
    "yolov10n": str(ROOT_DIR / "bestyolov10n.pt"),
    "yolov10m": str(ROOT_DIR / "bestyolov10m.pt"),
    "yolov11n": str(ROOT_DIR / "bestyolov11n.pt"),
    "yolov11m": str(ROOT_DIR / "bestyolov11m.pt"),
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device for model inference: {DEVICE}")


def load_test_data(data_yaml_path):
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    img_dir = Path(str(cfg["path"])) / cfg["val"]
    label_dir = img_dir.parent / "labels"
    img_paths = sorted([
        p for p in img_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    return img_paths, label_dir


def cache_model_predictions(checkpoints, img_paths, label_dir):
    """Run model inference once per model at low conf threshold to cache raw detections."""
    cache_file = OUTPUT_DIR / "raw_predictions_cache.pkl"
    import pickle
    
    if cache_file.exists():
        print(f"[INFO] Found existing prediction cache at {cache_file}. Loading...")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}. Re-running inference...")

    cached_data = {}
    
    for model_name, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            print(f"[WARN] Checkpoint {ckpt_path} not found! Skipping {model_name}.")
            continue
            
        print(f"[INFO] Caching predictions for {model_name} on {len(img_paths)} images (Device: {DEVICE})...", flush=True)
        model = YOLO(ckpt_path)
        img_records = []
        
        for idx, img_p in enumerate(img_paths):
            if (idx + 1) % 100 == 0 or (idx + 1) == len(img_paths):
                print(f"  -> {model_name}: {idx + 1}/{len(img_paths)} images processed...", flush=True)
            preds = model(img_p, imgsz=1024, device=DEVICE, verbose=False, iou=0.5, conf=0.05)[0]
            orig_h, orig_w = preds.orig_shape
            
            # Ground truth
            label_p = label_dir / f"{img_p.stem}.txt"
            gt_boxes = parse_yolo_labels(label_p, (orig_w, orig_h))
            
            # Raw boxes
            raw_boxes = defaultdict(list)
            if preds.boxes is not None:
                for box in preds.boxes:
                    conf = float(box.conf.item())
                    cls_id = int(box.cls.item())
                    xyxy = box.xyxy[0].tolist()
                    raw_boxes[cls_id].append(tuple(xyxy) + (conf,))
                    
            img_records.append({
                "img_name": img_p.name,
                "orig_size": (orig_w, orig_h),
                "raw_boxes": dict(raw_boxes),
                "gt_boxes": gt_boxes,
            })
            
        cached_data[model_name] = img_records
        
    with open(cache_file, "wb") as f:
        pickle.dump(cached_data, f)
    print(f"[SUCCESS] Saved raw prediction cache to {cache_file}", flush=True)
    return cached_data



def filter_boxes_by_conf(raw_boxes_dict, conf_thresh):
    filtered = defaultdict(list)
    for cls_id, b_list in raw_boxes_dict.items():
        for b in b_list:
            if b[4] >= conf_thresh:
                filtered[cls_id].append(b)
    return dict(filtered)


# ─── 1. Instance-Level vs Class-Level CCS ──────────────────────────────────

def run_instance_vs_class_experiment(cached_data, sem_matrix):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Instance-Level (Hungarian) vs. Class-Level CCS")
    print("="*70)
    
    results = {}
    
    for model_name, img_records in cached_data.items():
        class_ccs_list, class_csp_list, class_csem_list = [], [], []
        inst_ccs_list, inst_csp_list, inst_csem_list = [], [], []
        
        for rec in img_records:
            ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=0.25)
            gt_boxes = rec["gt_boxes"]
            
            # Class-level
            res_cls = compute_ccs(ai_boxes, gt_boxes, sem_matrix, alpha=0.5, beta=0.5)
            class_ccs_list.append(res_cls["ccs"])
            class_csp_list.append(res_cls["c_sp"])
            class_csem_list.append(res_cls["c_sem"])
            
            # Instance-level Hungarian
            res_inst = compute_ccs_instance_hungarian(ai_boxes, gt_boxes, sem_matrix, alpha=0.5, beta=0.5)
            inst_ccs_list.append(res_inst["ccs"])
            inst_csp_list.append(res_inst["c_sp"])
            inst_csem_list.append(res_inst["c_sem"])
            
        results[model_name] = {
            "class_level": {
                "mean_ccs": float(np.mean(class_ccs_list)),
                "std_ccs": float(np.std(class_ccs_list)),
                "mean_csp": float(np.mean(class_csp_list)),
                "mean_csem": float(np.mean(class_csem_list)),
                "scores": class_ccs_list,
            },
            "instance_level": {
                "mean_ccs": float(np.mean(inst_ccs_list)),
                "std_ccs": float(np.std(inst_ccs_list)),
                "mean_csp": float(np.mean(inst_csp_list)),
                "mean_csem": float(np.mean(inst_csem_list)),
                "scores": inst_ccs_list,
            }
        }
        
        print(f"[{model_name:<9}] Class-CCS: {np.mean(class_ccs_list):.4f} (Csp: {np.mean(class_csp_list):.4f}, Csem: {np.mean(class_csem_list):.4f}) | "
              f"Inst-CCS: {np.mean(inst_ccs_list):.4f} (Csp: {np.mean(inst_csp_list):.4f}, Csem: {np.mean(inst_csem_list):.4f})")
        
    return results


# ─── 2. Confidence Threshold Sweep & Area Under CCS Curve ──────────────────

def run_confidence_sweep_experiment(cached_data, sem_matrix):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Confidence Threshold Sweep (0.10 to 0.90) & mCCS")
    print("="*70)
    
    thresholds = np.linspace(0.10, 0.90, 17)
    sweep_results = {}
    
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    color_map = {
        "yolov8n": "#1f77b4", "yolov8m": "#ff7f0e",
        "yolov10n": "#2ca02c", "yolov10m": "#d62728",
        "yolov11n": "#9467bd", "yolov11m": "#8c564b",
    }
    
    for model_name, img_records in cached_data.items():
        ccs_means = []
        inst_ccs_means = []
        
        for t in thresholds:
            t_val = float(t)
            cur_cls_ccs = []
            cur_inst_ccs = []
            for rec in img_records:
                ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=t_val)
                gt_boxes = rec["gt_boxes"]
                r_cls = compute_ccs(ai_boxes, gt_boxes, sem_matrix)
                r_inst = compute_ccs_instance_hungarian(ai_boxes, gt_boxes, sem_matrix)
                cur_cls_ccs.append(r_cls["ccs"])
                cur_inst_ccs.append(r_inst["ccs"])
            ccs_means.append(float(np.mean(cur_cls_ccs)))
            inst_ccs_means.append(float(np.mean(cur_inst_ccs)))
            
        mccs = float(np.mean(ccs_means))
        trapz_fn = getattr(np, "trapezoid", np.trapz)
        auc_ccs = float(trapz_fn(ccs_means, thresholds) / (0.90 - 0.10))
        mccs_inst = float(np.mean(inst_ccs_means))

        
        sweep_results[model_name] = {
            "thresholds": [float(x) for x in thresholds],
            "class_ccs_curve": ccs_means,
            "instance_ccs_curve": inst_ccs_means,
            "mCCS": mccs,
            "AUC_CCS": auc_ccs,
            "mCCS_instance": mccs_inst,
        }
        
        print(f"[{model_name:<9}] mCCS: {mccs:.4f} | AUC_CCS: {auc_ccs:.4f} | mCCS_inst: {mccs_inst:.4f}")
        
        plt.plot(
            thresholds, ccs_means,
            label=f"{model_name} (mCCS = {mccs:.4f})",
            color=color_map.get(model_name, None),
            linewidth=2.2,
            marker='o', markersize=4
        )
        
    plt.title("Continuous Concordance Score (CCS) across Confidence Thresholds", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Detection Confidence Threshold ($\\tau_{\\mathrm{conf}}$)", fontsize=11)
    plt.ylabel("Continuous Concordance Score (CCS)", fontsize=11)
    plt.ylim(0.45, 0.70)
    plt.axvline(0.25, color="gray", linestyle="--", alpha=0.7, label="Default Operating Point (0.25)")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="lower left")
    plt.tight_layout()
    
    fig_path1 = OUTPUT_DIR / "fig_confidence_sweep.pdf"
    fig_path2 = SIVP_DIR / "fig_confidence_sweep.pdf"
    plt.savefig(fig_path1)
    plt.savefig(fig_path2)
    plt.close()
    print(f"[INFO] Saved confidence sweep plot to {fig_path1} and {fig_path2}")
    
    return sweep_results


# ─── 3. Taxonomy Sensitivity Analysis ──────────────────────────────────────

def run_taxonomy_sensitivity_experiment(cached_data):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Taxonomy Sensitivity Analysis across 4 Hierarchies")
    print("="*70)
    
    taxonomies = {
        "Default (3-branch)": TAXONOMY_TREE,
        "Alt1 (4-branch, Indep. Peeled)": TAXONOMY_TREE_ALT1,
        "Alt2 (Syndromic Pathological)": TAXONOMY_TREE_ALT2,
        "Alt3 (2-level Flat Body/Coating)": TAXONOMY_TREE_ALT3,
    }
    
    matrices = {
        tax_name: build_semantic_matrix_from_tree(tree)
        for tax_name, tree in taxonomies.items()
    }
    
    results = {}
    model_names = list(cached_data.keys())
    
    for tax_name, sem_mat in matrices.items():
        model_scores = {}
        for m_name in model_names:
            img_records = cached_data[m_name]
            scores = []
            for rec in img_records:
                ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=0.25)
                res = compute_ccs(ai_boxes, rec["gt_boxes"], sem_mat)
                scores.append(res["ccs"])
            model_scores[m_name] = float(np.mean(scores))
        results[tax_name] = model_scores
        
    print(f"{'Model':<10}" + "".join(f"{k[:18]:<22}" for k in taxonomies.keys()))
    print("-" * 95)
    for m_name in model_names:
        row = f"{m_name:<10}" + "".join(f"{results[k][m_name]:<22.4f}" for k in taxonomies.keys())
        print(row)
        
    # Check rank correlations between Default and alternatives
    default_ranks = [results["Default (3-branch)"][m] for m in model_names]
    rank_corrs = {}
    for tax_name in taxonomies.keys():
        alt_ranks = [results[tax_name][m] for m in model_names]
        rho, p_val = stats.spearmanr(default_ranks, alt_ranks)
        tau, p_tau = stats.kendalltau(default_ranks, alt_ranks)
        rank_corrs[tax_name] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
        }
        print(f"Rank Correlation with Default [{tax_name}]: Spearman rho = {rho:.4f} (p={p_val:.4f}), Kendall tau = {tau:.4f}")
        
    return {"scores": results, "correlations": rank_corrs}


# ─── 4. Computational Latency & Runtime Benchmark ──────────────────────────

def run_runtime_benchmark(cached_data, sem_matrix):
    print("\n" + "="*70)
    print("EXPERIMENT 4: Computational Latency Benchmark on CPU")
    print("="*70)
    
    # Use first model cached records for fair comparison across 551 images
    records = cached_data["yolov8m"]
    N_images = len(records)
    
    # 1. IoU Greedy Class Matching
    t0 = time.perf_counter()
    for rec in records:
        ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=0.25)
        gt_boxes = rec["gt_boxes"]
        classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
        for cls_id in classes:
            ai_list = [b[:4] for b in ai_boxes.get(cls_id, [])]
            gt_list = gt_boxes.get(cls_id, [])
            for a_box in ai_list:
                for g_box in gt_list:
                    # Compute standard IoU
                    x1 = max(a_box[0], g_box[0]); y1 = max(a_box[1], g_box[1])
                    x2 = min(a_box[2], g_box[2]); y2 = min(a_box[3], g_box[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area_a = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
                    area_b = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                    _ = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0
    t_iou = time.perf_counter() - t0
    
    # 2. Class-Level CCS
    t0 = time.perf_counter()
    for rec in records:
        ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=0.25)
        _ = compute_ccs(ai_boxes, rec["gt_boxes"], sem_matrix)
    t_class_ccs = time.perf_counter() - t0
    
    # 3. Full COCO mAP@[.5:.95] across 10 IoU thresholds
    t0 = time.perf_counter()
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    for rec in records:
        ai_boxes = rec["raw_boxes"]
        gt_boxes = rec["gt_boxes"]
        for thresh in iou_thresholds:
            for cls_id in set(ai_boxes.keys()) | set(gt_boxes.keys()):
                ai_l = sorted([b for b in ai_boxes.get(cls_id, []) if b[4] >= 0.001], key=lambda x: x[4], reverse=True)
                gt_l = gt_boxes.get(cls_id, [])
                assigned = set()
                for a in ai_l:
                    for g_idx, g in enumerate(gt_l):
                        if g_idx not in assigned:
                            x1 = max(a[0], g[0]); y1 = max(a[1], g[1])
                            x2 = min(a[2], g[2]); y2 = min(a[3], g[3])
                            inter = max(0, x2 - x1) * max(0, y2 - y1)
                            area_a = (a[2] - a[0]) * (a[3] - a[1])
                            area_b = (g[2] - g[0]) * (g[3] - g[1])
                            iou_val = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0
                            if iou_val >= thresh:
                                assigned.add(g_idx)
                                break
    t_coco_map = time.perf_counter() - t0

    # 4. Instance-Level Hungarian CCS
    t0 = time.perf_counter()
    for rec in records:
        ai_boxes = filter_boxes_by_conf(rec["raw_boxes"], conf_thresh=0.25)
        _ = compute_ccs_instance_hungarian(ai_boxes, rec["gt_boxes"], sem_matrix)
    t_inst_ccs = time.perf_counter() - t0
    
    def to_1k(elapsed_sec, n_img):
        sec_1k = (elapsed_sec / n_img) * 1000.0
        ms_img = (elapsed_sec / n_img) * 1000.0
        fps = n_img / elapsed_sec if elapsed_sec > 0 else 0.0
        return sec_1k, ms_img, fps
    
    sec_iou, ms_iou, fps_iou = to_1k(t_iou, N_images)
    sec_class, ms_class, fps_class = to_1k(t_class_ccs, N_images)
    sec_map, ms_map, fps_map = to_1k(t_coco_map, N_images)
    sec_inst, ms_inst, fps_inst = to_1k(t_inst_ccs, N_images)
    
    benchmark_res = {
        "num_images_tested": N_images,
        "standard_iou": {"sec_per_1000_img": sec_iou, "ms_per_image": ms_iou, "fps": fps_iou},
        "class_level_ccs": {"sec_per_1000_img": sec_class, "ms_per_image": ms_class, "fps": fps_class},
        "coco_map_50_95": {"sec_per_1000_img": sec_map, "ms_per_image": ms_map, "fps": fps_map},
        "instance_level_hungarian_ccs": {"sec_per_1000_img": sec_inst, "ms_per_image": ms_inst, "fps": fps_inst},
    }
    
    print(f"Standard IoU Matching       : {ms_iou:.3f} ms/image ({fps_iou:.0f} FPS | {sec_iou:.3f} s / 1k images)")
    print(f"Class-Level CCS (Closed Form): {ms_class:.3f} ms/image ({fps_class:.0f} FPS | {sec_class:.3f} s / 1k images)")
    print(f"Full COCO mAP@[.5:.95]       : {ms_map:.3f} ms/image ({fps_map:.0f} FPS | {sec_map:.3f} s / 1k images)")
    print(f"Instance Hungarian CCS      : {ms_inst:.3f} ms/image ({fps_inst:.0f} FPS | {sec_inst:.3f} s / 1k images)")

    
    return benchmark_res


# ─── 5. Statistical Effect Sizes (Cohen's d & Hedges' g) ───────────────────

def run_effect_sizes_experiment(exp1_results):
    print("\n" + "="*70)
    print("EXPERIMENT 5: Statistical Effect Sizes (Cohen's d and Hedges' g)")
    print("="*70)
    
    models = list(exp1_results.keys())
    pairwise_stats = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            s_a = np.array(exp1_results[m_a]["class_level"]["scores"])
            s_b = np.array(exp1_results[m_b]["class_level"]["scores"])
            diff = s_a - s_b
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            
            n = len(diff)
            cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
            # Hedges' g correction factor
            j_corr = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
            hedges_g = cohen_d * j_corr
            
            # Wilcoxon test excluding ties
            non_zero_diff = diff[diff != 0]
            if len(non_zero_diff) > 0:
                w_stat, w_pval = stats.wilcoxon(non_zero_diff)
            else:
                w_stat, w_pval = 0.0, 1.0
                
            pairwise_stats.append({
                "model_A": m_a,
                "model_B": m_b,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "cohen_d": float(cohen_d),
                "hedges_g": float(hedges_g),
                "wilcoxon_p": float(w_pval),
                "bonferroni_sig": bool(w_pval < (0.05 / 15.0)),
            })
            
            print(f"[{m_a:<8} vs {m_b:<8}] MeanDiff: {mean_diff:+.4f} | Cohen's d: {cohen_d:+.4f} | Hedges' g: {hedges_g:+.4f} | Wilcoxon p: {w_pval:.4e} {'*' if w_pval < (0.05/15) else ''}")
            
    return pairwise_stats


# ─── Main Execution ─────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STARTING FULL REBUTTAL EXPERIMENTS SUITE FOR CCS")
    print("=" * 80)
    
    img_paths, label_dir = load_test_data(DATA_YAML)
    print(f"[INFO] Loaded {len(img_paths)} test images from {DATA_YAML}")
    
    # 0. Build default semantic matrix
    sem_matrix = build_semantic_matrix()
    
    # Cache predictions
    cached_data = cache_model_predictions(CHECKPOINTS, img_paths, label_dir)
    
    # Run all 5 experiments
    exp1_res = run_instance_vs_class_experiment(cached_data, sem_matrix)
    exp2_res = run_confidence_sweep_experiment(cached_data, sem_matrix)
    exp3_res = run_taxonomy_sensitivity_experiment(cached_data)
    exp4_res = run_runtime_benchmark(cached_data, sem_matrix)
    exp5_res = run_effect_sizes_experiment(exp1_res)
    
    # Aggregate and save results
    rebuttal_payload = {
        "experiment1_instance_vs_class": {
            m: {
                "class_level": {k: v for k, v in data["class_level"].items() if k != "scores"},
                "instance_level": {k: v for k, v in data["instance_level"].items() if k != "scores"},
            }
            for m, data in exp1_res.items()
        },
        "experiment2_confidence_sweep": exp2_res,
        "experiment3_taxonomy_sensitivity": exp3_res,
        "experiment4_runtime_benchmark": exp4_res,
        "experiment5_effect_sizes": exp5_res,
    }
    
    out_json = OUTPUT_DIR / "rebuttal_results.json"
    with open(out_json, "w") as f:
        json.dump(rebuttal_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved comprehensive rebuttal results to {out_json}")


if __name__ == "__main__":
    main()
