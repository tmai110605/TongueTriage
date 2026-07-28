"""
map_metrics.py — COCO-style mAP@[0.5:0.95] baseline for CCS Paper Experiments v2.

Đây là baseline quan trọng vì mAP@[0.5:0.95] (trung bình AP trên 10 ngưỡng IoU
0.50, 0.55, ..., 0.95) chính là giải pháp chuẩn của cộng đồng detection cho
đúng vấn đề "threshold sensitivity" mà CCS đang dùng để lập luận. Nếu bài báo
không so sánh với mAP@[0.5:0.95], reviewer sẽ hỏi ngay "sao không dùng luôn
COCO mAP?" — module này giúp trả lời câu đó bằng số liệu thực tế.

KHÔNG sửa gì trong ccs.py. Chỉ cần import 3 hàm _compute_iou/_giou/_diou đã có
sẵn trong experiments_ccs_v2.py (không viết lại) và dùng lại cấu trúc per_image
mà _eval_one_model() đã tạo ra.

Cách tích hợp: xem 3 chỗ sửa trong hướng dẫn kèm theo (không lặp lại ở đây).
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ─── AP core (COCO 101-point interpolation) ─────────────────────────────────

def _compute_ap_101point(recalls, precisions):
    """AP nội suy 101 điểm, đúng chuẩn COCO eval (pycocotools)."""
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)

    # ép precision đơn điệu giảm từ phải sang trái (envelope chuẩn PR curve)
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
    AP cho MỘT lớp, ở MỘT ngưỡng IoU/GIoU/DIoU cụ thể, gộp trên toàn bộ ảnh.

    dets: list các tuple (img_idx, box[x1,y1,x2,y2], score) — PHẢI là tập đầy đủ,
          KHÔNG lọc theo confidence trước, nếu không đường cong PR sẽ bị cắt cụt
          và AP tính ra sẽ thấp hơn thực tế.
    gts_by_img: dict img_idx -> list các GT box [x1,y1,x2,y2] của lớp đó.
    sim_fn: một trong _compute_iou / _compute_giou / _compute_diou.
    """
    total_gt = sum(len(v) for v in gts_by_img.values())
    if total_gt == 0:
        return None  # lớp không xuất hiện trong GT -> AP không xác định (quy ước COCO: bỏ qua)

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
    Tính mAP@[.5:.95], mAP@.5, mAP@.75 cho MỘT model, theo metric IoU/GIoU/DIoU.

    per_image: list dict do _eval_one_model() trả về (đã cache sẵn).
    metric_fns: dict {"iou": _compute_iou, "giou": _compute_giou, "diou": _compute_diou}
                (truyền lại 3 hàm đã có sẵn trong experiments_ccs_v2.py, không viết lại).
    box_field: key chứa detection CHƯA lọc confidence — mặc định "ai_boxes_map"
               (xem hướng dẫn sửa _eval_one_model để tạo field này).
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


# ─── Experiment wrapper (giữ đúng style print/save như các exp_* khác) ──────

def exp_map_comparison(all_results, metric_fns, output_dir):
    """
    Chạy compute_map_5095 cho mọi model x {iou, giou, diou}, in bảng, lưu
    map_5095.json, và trả về rows để main() gộp thêm vào summary_ranking.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: COCO-style mAP@[0.5:0.95] (IoU / GIoU / DIoU) vs CCS")
    print("=" * 70)

    rows = []
    per_class_rows = []  # giữ lại breakdown theo từng lớp trong 8 lớp, không vứt đi
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