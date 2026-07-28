"""
nwd_baseline.py — Normalized Wasserstein Distance (NWD) baseline.

Tài liệu tham khảo: Wang, J. et al. (2021), "A Normalized Gaussian Wasserstein
Distance for Tiny Object Detection", arXiv:2110.13389.

TẠI SAO CẦN BASELINE NÀY: NWD cũng model bounding box thành Gaussian 2D
giống C_sp, nhưng dùng khoảng cách Wasserstein thay vì cosine similarity
chuẩn hóa. Đây là "họ hàng gần nhất" của C_sp trong literature — nếu bài báo
không so sánh, reviewer quen NWD/GWD/KLD sẽ hỏi ngay "khác gì NWD?". Module
này giúp trả lời bằng số liệu: (1) F1 theo threshold dùng NWD làm metric
matching, y hệt cấu trúc IoU/GIoU/DIoU đã có, và (2) một "NWD liên tục"
(không ngưỡng) tính song song C_sp để so sánh trực tiếp + đo tương quan.

Quy ước box của NWD gốc: R=(cx,cy,w,h) -> Gaussian N(mu, Sigma) với
mu=(cx,cy), Sigma=diag(w^2/4, h^2/4) (KHÁC với sigma=(max-min)/4 mà C_sp
trong ccs.py dùng — đây là 2 cách model hóa khác nhau, đúng theo bài gốc
của mỗi phương pháp, không phải lỗi).
"""

import math
import json
from pathlib import Path

import numpy as np


# ─── NWD core ─────────────────────────────────────────────────────────────

def _compute_nwd(box_a, box_b, C):
    """
    NWD giữa 2 box [x1,y1,x2,y2], theo đúng công thức Wang et al. 2021.
    Trả về giá trị trong (0,1], 1 = trùng khớp hoàn toàn.
    C: hằng số chuẩn hóa phụ thuộc dataset (>0), xem estimate_nwd_constant().
    """
    cx_a, cy_a = (box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2
    w_a, h_a = box_a[2] - box_a[0], box_a[3] - box_a[1]
    cx_b, cy_b = (box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2
    w_b, h_b = box_b[2] - box_b[0], box_b[3] - box_b[1]

    w2_sq = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
              + ((w_a - w_b) / 2) ** 2 + ((h_a - h_b) / 2) ** 2)
    return math.exp(-math.sqrt(max(w2_sq, 0.0)) / C)


def estimate_nwd_constant(per_image_any_model):
    """
    C = trung bình sqrt(w*h) của toàn bộ GT box trong tập test — đúng tinh
    thần bài gốc (họ dùng "average absolute size" của dataset AI-TOD, KHÔNG
    dùng giá trị hardcode của dataset khác). GT giống nhau ở mọi model nên
    chỉ cần truyền per_image của 1 model bất kỳ.
    """
    sizes = []
    for img in per_image_any_model:
        for boxes in img["gt_boxes"].values():
            for b in boxes:
                w, h = b[2] - b[0], b[3] - b[1]
                if w > 0 and h > 0:
                    sizes.append(math.sqrt(w * h))
    return float(np.mean(sizes)) if sizes else 1.0


# ─── Threshold-based matching bằng NWD (song song IoU/GIoU/DIoU) ───────────

def _match_boxes_nwd(ai_list, gt_list, thresh, C):
    unmatched_gt = list(range(len(gt_list)))
    tp = fp = 0
    ai_order = sorted(range(len(ai_list)), key=lambda i: ai_list[i][4], reverse=True)
    for ai_idx in ai_order:
        a_box = ai_list[ai_idx][:4]
        best_sim, best_gt_idx = -1.0, -1
        for gt_idx in unmatched_gt:
            sim = _compute_nwd(a_box, gt_list[gt_idx], C)
            if sim > best_sim:
                best_sim, best_gt_idx = sim, gt_idx
        if best_gt_idx != -1 and best_sim >= thresh:
            tp += 1
            unmatched_gt.remove(best_gt_idx)
        else:
            fp += 1
    fn = len(unmatched_gt)
    return tp, fp, fn


def _accumulate_nwd(per_image, thresh, C):
    tp = fp = fn = 0
    for img in per_image:
        ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
        classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
        for cls_id in classes:
            t, f, n = _match_boxes_nwd(ai_boxes.get(cls_id, []), gt_boxes.get(cls_id, []), thresh, C)
            tp += t
            fp += f
            fn += n
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ─── NWD liên tục (không ngưỡng) — so sánh trực tiếp với C_sp ──────────────

def continuous_nwd_per_image(per_image, C):
    """
    Với mỗi ảnh: với mỗi nhãn ở AI hoặc GT, lấy NWD của cặp khớp tốt nhất
    (quy ước 0 nếu nhãn chỉ có ở một phía) rồi trung bình trên ảnh — cùng
    quy ước như C_sp trong ccs.py, để so sánh táo với táo.
    """
    scores = []
    for img in per_image:
        ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
        labels = set(ai_boxes.keys()) | set(gt_boxes.keys())
        if not labels:
            continue
        vals = []
        for k in labels:
            a_list, g_list = ai_boxes.get(k, []), gt_boxes.get(k, [])
            if not a_list or not g_list:
                vals.append(0.0)
                continue
            best = -1.0
            for a in a_list:
                for g in g_list:
                    s = _compute_nwd(a[:4], g, C)
                    if s > best:
                        best = s
            vals.append(best)
        scores.append(float(np.mean(vals)))
    return scores


def _pearson_r(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / denom) if denom > 0 else 0.0


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_nwd_comparison(all_results, output_dir, thresholds=(0.3, 0.5, 0.7)):
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: NWD Baseline (Wang et al. 2021) vs C_sp / CCS")
    print("=" * 70)

    any_model = next(iter(all_results.values()))["per_image"]
    C = estimate_nwd_constant(any_model)
    print(f"\n  NWD constant C (avg sqrt(GT area)) = {C:.4f}")

    rows = []
    for model_name, data in all_results.items():
        per_image = data["per_image"]
        c_sp_vals = [img["ccs"]["c_sp"] for img in per_image]
        nwd_vals = continuous_nwd_per_image(per_image, C)

        n = min(len(c_sp_vals), len(nwd_vals))
        corr = _pearson_r(c_sp_vals[:n], nwd_vals[:n])

        row = {
            "model": model_name,
            "mean_C_sp": round(float(np.mean(c_sp_vals)), 4),
            "mean_NWD_continuous": round(float(np.mean(nwd_vals)), 4),
            "pearson_r_Csp_vs_NWD": round(corr, 4),
        }
        for t in thresholds:
            s = _accumulate_nwd(per_image, t, C)
            row[f"F1_NWD@{t}"] = s["f1"]
        rows.append(row)

    header = (f"{'Model':<10} {'C_sp':<8} {'NWD':<8} {'r(Csp,NWD)':<12} "
              + " ".join(f"F1_NWD@{t:<6}" for t in thresholds))
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        line = f"{r['model']:<10} {r['mean_C_sp']:<8.4f} {r['mean_NWD_continuous']:<8.4f} {r['pearson_r_Csp_vs_NWD']:<12.4f}"
        for t in thresholds:
            line += f" {r[f'F1_NWD@{t}']:<9.4f}"
        print(line)

    print("\n  Diễn giải: r(C_sp, NWD) cao (>0.9) nghĩa là hai cách model hóa Gaussian")
    print("  (cosine similarity vs Wasserstein distance) cho thứ hạng độ khớp không gian")
    print("  gần như nhau trên dữ liệu này -> phần đóng góp mới của CCS chủ yếu nằm ở")
    print("  C_sem (ngữ nghĩa), không phải cách đo C_sp. Nếu r thấp, cần giải thích vì sao.")

    save_path = Path(output_dir) / "nwd_comparison.json"
    with open(save_path, "w") as f:
        json.dump({"nwd_constant_C": round(C, 4), "rows": rows}, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return rows
