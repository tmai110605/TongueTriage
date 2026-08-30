"""
Compute 1,000-iteration Bootstrap Confidence Intervals (Mean +- Std and 95% CI)
for all 6 models under both Class-level CCS and Instance-level Hungarian CCS.
"""

import json
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from ccs import compute_ccs, compute_ccs_instance_hungarian, build_semantic_matrix

OUTPUT_DIR = ROOT_DIR / "runs" / "experiments"

def main():
    cache_file = OUTPUT_DIR / "raw_predictions_cache.pkl"
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
        
    sem_matrix = build_semantic_matrix()
    
    np.random.seed(42)
    n_bootstraps = 1000
    n_images = len(cached_data["yolov8n"])
    
    # 1. Compute per-image scores for each model
    per_img_class_ccs = {}
    per_img_inst_ccs = {}
    
    for model_name, records in cached_data.items():
        c_scores = []
        i_scores = []
        for rec in records:
            # apply standard conf 0.25, nms 0.5
            ai_boxes = {}
            for cid, blist in rec["raw_boxes"].items():
                vb = [b for b in blist if b[4] >= 0.25]
                if vb:
                    ai_boxes[cid] = vb
            
            c_res = compute_ccs(ai_boxes, rec["gt_boxes"], sem_matrix)
            i_res = compute_ccs_instance_hungarian(ai_boxes, rec["gt_boxes"], sem_matrix)
            
            c_scores.append(c_res["ccs"])
            i_scores.append(i_res["ccs"])
            
        per_img_class_ccs[model_name] = np.array(c_scores)
        per_img_inst_ccs[model_name] = np.array(i_scores)
        
    # 2. Bootstrap sampling
    results = {}
    print("\n" + "="*80)
    print(f"BOOTSTRAP RESAMPLING ANALYSIS (N = {n_images} images, {n_bootstraps} iterations)")
    print("="*80)
    
    for model_name in cached_data.keys():
        c_arr = per_img_class_ccs[model_name]
        i_arr = per_img_inst_ccs[model_name]
        
        c_boot_means = []
        i_boot_means = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(n_images, size=n_images, replace=True)
            c_boot_means.append(np.mean(c_arr[indices]))
            i_boot_means.append(np.mean(i_arr[indices]))
            
        c_boot_means = np.array(c_boot_means)
        i_boot_means = np.array(i_boot_means)
        
        c_mean = float(np.mean(c_arr))
        c_std = float(np.std(c_boot_means))
        c_ci_low = float(np.percentile(c_boot_means, 2.5))
        c_ci_high = float(np.percentile(c_boot_means, 97.5))
        
        i_mean = float(np.mean(i_arr))
        i_std = float(np.std(i_boot_means))
        i_ci_low = float(np.percentile(i_boot_means, 2.5))
        i_ci_high = float(np.percentile(i_boot_means, 97.5))
        
        results[model_name] = {
            "class_ccs": {
                "mean": c_mean,
                "bootstrap_std": c_std,
                "ci_95": [c_ci_low, c_ci_high]
            },
            "instance_ccs": {
                "mean": i_mean,
                "bootstrap_std": i_std,
                "ci_95": [i_ci_low, i_ci_high]
            }
        }
        
        print(f"[{model_name:<9}] Class CCS   : {c_mean:.4f} ± {c_std:.4f}  (95% CI: [{c_ci_low:.4f}, {c_ci_high:.4f}])")
        print(f"[{model_name:<9}] Instance CCS: {i_mean:.4f} ± {i_std:.4f}  (95% CI: [{i_ci_low:.4f}, {i_ci_high:.4f}])")
        print("-" * 80)
        
    out_file = OUTPUT_DIR / "bootstrap_ci_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SUCCESS] Saved bootstrap results to {out_file}")

if __name__ == "__main__":
    main()
