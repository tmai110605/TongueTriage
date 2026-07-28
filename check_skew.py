"""
check_skew.py — Check the "skewed/long-tail distribution" hypothesis for the 3 pairs
where Wilcoxon and bootstrap gave discordant results.

USAGE: run directly `python check_skew.py`
"""
from scipy.stats import skew, kurtosis
import numpy as np

# Import necessary functions from experiments_ccs_v2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments_ccs_v2 import (
    available_checkpoints,
    load_test_paths,
    _eval_one_model,
    build_semantic_matrix,
    DATA_YAML
)

def check_pair(name, diffs):
    diffs = np.asarray(diffs)
    print(f"\n{name}  (n={len(diffs)})")
    print(f"  mean={diffs.mean():.4f}  median={np.median(diffs):.4f}  std={diffs.std():.4f}")
    print(f"  skewness={skew(diffs):.3f}  excess kurtosis={kurtosis(diffs):.3f}")
    
    pct_zero = 100 * (np.abs(diffs) < 1e-6).mean()
    print(f"  % zero diff={pct_zero:.1f}% (images with CCS difference = 0)")
    
    # |skewness| > 1 or kurtosis > 3-4 is usually considered significantly skewed/long-tailed
    flag = abs(skew(diffs)) > 1 or kurtosis(diffs) > 3
    print(f"  -> {'SIGNIFICANT skew/long-tail detected' if flag else 'No clear skewness detected'}")

def main():
    print("Loading data and config...")
    sem_matrix = build_semantic_matrix()
    checkpoints = available_checkpoints()
    img_paths, label_dir = load_test_paths(DATA_YAML)
    
    models_to_run = ["yolov8n", "yolov8m", "yolov11n", "yolov10n"]
    results = {}
    
    print("\nStarting to extract per-image CCS scores for the 4 required models...")
    for model_name in models_to_run:
        if model_name not in checkpoints:
            print(f"Warning: Checkpoint for {model_name} not found")
            continue
            
        print(f" Running inference for {model_name}...")
        per_image = _eval_one_model(checkpoints, img_paths, label_dir, sem_matrix, model_name)
        results[model_name] = np.array([img["ccs"]["ccs"] for img in per_image])
        
    print("\n" + "="*50)
    print("SKEWNESS CHECK RESULTS")
    print("="*50)
    
    if "yolov8n" in results and "yolov8m" in results:
        check_pair("YOLOv8n vs YOLOv8m", results["yolov8n"] - results["yolov8m"])
        
    if "yolov8n" in results and "yolov11n" in results:
        check_pair("YOLOv8n vs YOLO11n", results["yolov8n"] - results["yolov11n"])
        
    if "yolov8m" in results and "yolov10n" in results:
        check_pair("YOLOv8m vs YOLOv10n", results["yolov8m"] - results["yolov10n"])

if __name__ == "__main__":
    main()
