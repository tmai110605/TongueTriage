"""
Benchmark exact evaluation time for:
  1. Full COCO mAP@[.5:.95] (PR curve integration across 10 IoU thresholds)
  2. Standard IoU Greedy Matching (F1@0.5)
  3. Class-Level CCS (Closed-Form)
  4. Instance-Level Hungarian CCS
on a standard CPU (seconds per 1,000 images and ms per image).
"""

import time
import pickle
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from ccs import compute_ccs, compute_ccs_instance_hungarian, build_semantic_matrix

OUTPUT_DIR = ROOT_DIR / "runs" / "experiments"

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

def benchmark_all():
    cache_file = OUTPUT_DIR / "raw_predictions_cache.pkl"
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
        
    records = cached_data["yolov8m"]
    n_images = len(records)
    n_runs = 5
    
    sem_matrix = build_semantic_matrix()
    
    # 1. Benchmark Standard IoU Matching (F1@0.5)
    iou_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for rec in records:
            ai_boxes = rec["raw_boxes"]
            gt_boxes = rec["gt_boxes"]
            tp = fp = fn = 0
            for cls_id in set(ai_boxes.keys()) | set(gt_boxes.keys()):
                ai_l = [b[:4] for b in ai_boxes.get(cls_id, []) if b[4] >= 0.25]
                gt_l = gt_boxes.get(cls_id, [])
                matched = 0
                for a in ai_l:
                    if any(_compute_iou(a, g) >= 0.5 for g in gt_l):
                        matched += 1
                tp += matched
                fp += len(ai_l) - matched
                fn += max(0, len(gt_l) - matched)
        iou_times.append(time.perf_counter() - t0)
        
    # 2. Benchmark Full COCO mAP@[.5:.95] across 10 IoU thresholds
    map_times = []
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        # compute matches across all 10 thresholds and sort PR points
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
                            if g_idx not in assigned and _compute_iou(a[:4], g) >= thresh:
                                assigned.add(g_idx)
                                break
        map_times.append(time.perf_counter() - t0)

    # 3. Benchmark Class-Level Closed-Form CCS
    ccs_class_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for rec in records:
            ai_boxes = {cid: [b for b in bl if b[4] >= 0.25] for cid, bl in rec["raw_boxes"].items()}
            compute_ccs(ai_boxes, rec["gt_boxes"], sem_matrix)
        ccs_class_times.append(time.perf_counter() - t0)

    # 4. Benchmark Instance-Level Hungarian CCS
    ccs_inst_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for rec in records:
            ai_boxes = {cid: [b for b in bl if b[4] >= 0.25] for cid, bl in rec["raw_boxes"].items()}
            compute_ccs_instance_hungarian(ai_boxes, rec["gt_boxes"], sem_matrix)
        ccs_inst_times.append(time.perf_counter() - t0)
        
    print("\n" + "="*80)
    print("COMPUTATIONAL LATENCY BENCHMARK ON STANDARD CPU")
    print("="*80)
    
    def fmt(t_list):
        mean_t = np.mean(t_list)
        ms_per_img = (mean_t / n_images) * 1000
        sec_per_1k = (mean_t / n_images) * 1000
        return ms_per_img, sec_per_1k

    m_iou, s_iou = fmt(iou_times)
    m_map, s_map = fmt(map_times)
    m_ccs_c, s_ccs_c = fmt(ccs_class_times)
    m_ccs_i, s_ccs_i = fmt(ccs_inst_times)
    
    print(f"{'Metric':<35} | {'Latency (ms / img)':<20} | {'Time (s / 1,000 imgs)':<20}")
    print("-" * 80)
    print(f"{'Standard IoU Matching (F1@0.5)':<35} | {m_iou:18.2f} ms | {s_iou:18.2f} s")
    print(f"{'COCO mAP@[.5:.95] (10 IoU thresholds)':<35} | {m_map:18.2f} ms | {s_map:18.2f} s")
    print(f"{'Class-Level CCS (Closed-Form)':<35} | {m_ccs_c:18.2f} ms | {s_ccs_c:18.2f} s")
    print(f"{'Instance-Level Hungarian CCS':<35} | {m_ccs_i:18.2f} ms | {s_ccs_i:18.2f} s")
    print("=" * 80)
    
    return {
        "standard_iou_matching": {"ms_per_img": m_iou, "sec_per_1k": s_iou},
        "coco_map_50_95": {"ms_per_img": m_map, "sec_per_1k": s_map},
        "class_level_ccs": {"ms_per_img": m_ccs_c, "sec_per_1k": s_ccs_c},
        "instance_level_hungarian_ccs": {"ms_per_img": m_ccs_i, "sec_per_1k": s_ccs_i}
    }

if __name__ == "__main__":
    benchmark_all()
