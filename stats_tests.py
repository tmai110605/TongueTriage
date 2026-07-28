"""
stats_tests.py — Statistical significance testing for CCS Paper Experiments.

Giải quyết lỗ hổng: toàn bộ so sánh model trong v2 (flipped_pairs, summary_ranking)
hiện chỉ dựa trên GIÁ TRỊ TRUNG BÌNH điểm — không có khoảng tin cậy, không có
kiểm định. Với 6 model chênh nhau vài phần nghìn CCS (vd yolov10m 0.6413 vs
yolov8m 0.6421), không thể biết chênh lệch đó có ý nghĩa hay chỉ là nhiễu.

Module này dùng lại chính các ảnh trong test set làm đơn vị resampling (mỗi
ảnh độc lập), không cần train lại nhiều lần / không cần nhãn bác sĩ:

  1. bootstrap_score / paired_bootstrap_diff — bootstrap ở mức ảnh (image-level),
     cho CI 95% và p-value hai phía cho HIỆU của bất kỳ hàm điểm nào (CCS, F1, mAP...).
  2. wilcoxon_signed_rank — kiểm định phi tham số bắt cặp trên mảng điểm theo
     từng ảnh (vd CCS mỗi ảnh của model A vs model B) — không phụ thuộc scipy.
  3. spearman_rank_correlation — tương quan thứ hạng giữa 2 cách xếp hạng model
     (vd xếp theo CCS vs xếp theo F1@0.5) — không phụ thuộc scipy.

CẢNH BÁO ĐA KIỂM ĐỊNH (multiple comparisons): với 6 model có C(6,2)=15 cặp,
nếu kiểm định 15 lần ở alpha=0.05, xác suất có ít nhất 1 kết quả dương tính giả
tăng đáng kể. exp_significance() báo cáo cả p-value thô LẪN ngưỡng
Bonferroni-adjusted (alpha/15) — nên dùng ngưỡng đã hiệu chỉnh khi kết luận
trong bài báo.
"""

import math
import json
from pathlib import Path

import numpy as np


# ─── Bootstrap (image-level resampling) ─────────────────────────────────────

def bootstrap_score(per_image, score_fn, n_boot=2000, ci=0.95, seed=42):
    """
    CI 95% cho một hàm điểm bất kỳ score_fn(list_per_image_dict) -> float,
    bằng resample ảnh có hoàn lại.
    """
    rng = np.random.default_rng(seed)
    n = len(per_image)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled = [per_image[i] for i in idx]
        boot_vals[b] = score_fn(resampled)
    point = score_fn(per_image)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_vals, [alpha, 1 - alpha])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_diff(per_image_a, per_image_b, score_fn, n_boot=5000, seed=42):
    """
    per_image_a, per_image_b: CÙNG danh sách ảnh, CÙNG thứ tự (2 model chạy
    trên cùng test set) -> mỗi lần resample lấy CÙNG chỉ số ảnh cho cả 2 model
    (giữ đúng cấu trúc bắt cặp), rồi tính hiệu score_fn(A) - score_fn(B).

    score_fn: callable(list_per_image_dict) -> float. Có thể là mean CCS
    (rẻ) hoặc F1 tổng hợp (đắt hơn vì phải chạy lại matching) — với score_fn
    đắt, giảm n_boot xuống ~500-1000 để đỡ chậm.
    """
    assert len(per_image_a) == len(per_image_b), "Hai model phải chạy trên cùng danh sách ảnh, cùng thứ tự"
    n = len(per_image_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled_a = [per_image_a[i] for i in idx]
        resampled_b = [per_image_b[i] for i in idx]
        diffs[b] = score_fn(resampled_a) - score_fn(resampled_b)
    observed = score_fn(per_image_a) - score_fn(per_image_b)
    if observed >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    p = float(min(1.0, p))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "observed_diff": float(observed),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": p,
        "significant_0.05": bool(p < 0.05),
    }


# ─── Wilcoxon signed-rank (bắt cặp theo ảnh, tự cài đặt, không cần scipy) ──

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilcoxon_signed_rank(values_a, values_b):
    """
    Kiểm định Wilcoxon signed-rank (xấp xỉ chuẩn, không dùng scipy), cho 2
    mảng điểm CÙNG ĐỘ DÀI, bắt cặp theo chỉ số (vd CCS từng ảnh của model A/B).
    Cần n >= ~20 để xấp xỉ chuẩn đáng tin cậy (test set detection thường thừa).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return {"n_effective": int(n), "note": "n quá nhỏ (<10) để xấp xỉ chuẩn đáng tin cậy"}

    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    signed_ranks = ranks * np.sign(d)
    W_pos = signed_ranks[signed_ranks > 0].sum()
    W_neg = -signed_ranks[signed_ranks < 0].sum()
    W = min(W_pos, W_neg)

    mean_W = n * (n + 1) / 4.0
    std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mean_W) / std_W if std_W > 0 else 0.0
    p_value = min(1.0, 2 * (1 - _norm_cdf(abs(z))))

    return {
        "n_effective": int(n),
        "W_statistic": float(W),
        "z": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05),
    }


def per_image_f1_array(per_image, thresh, metric, match_boxes_for_class_fn):
    """
    F1 tính RIÊNG cho từng ảnh (không gộp cả tập trước khi tính), để có mảng
    bắt cặp dùng cho Wilcoxon. match_boxes_for_class_fn: truyền lại
    _match_boxes_for_class từ experiments_ccs_v2.py.
    LƯU Ý: F1 từng ảnh dễ bị 0/1 cực đoan (ảnh có ít box) -> nhiễu hơn CCS
    từng ảnh; Wilcoxon trên F1 từng ảnh nên xem là bổ sung, không thay cho
    kiểm định trên F1 tổng hợp (paired_bootstrap_diff).
    """
    f1s = []
    for img in per_image:
        ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
        classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
        tp = fp = fn = 0
        for cls_id in classes:
            t, f, n = match_boxes_for_class_fn(ai_boxes.get(cls_id, []), gt_boxes.get(cls_id, []), thresh, metric)
            tp += t
            fp += f
            fn += n
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return np.array(f1s)


# ─── Spearman rank correlation (tự cài đặt, không cần scipy) ───────────────

def _rank_with_ties(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(values_a, values_b):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    ra, rb = _rank_with_ties(a), _rank_with_ties(b)
    rac, rbc = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt((rac ** 2).sum() * (rbc ** 2).sum())
    return float((rac * rbc).sum() / denom) if denom > 0 else 0.0


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_significance(all_results, match_boxes_for_class_fn, n_boot_ccs=2000, n_boot_f1=800, output_dir="."):
    """
    Chạy toàn bộ pipeline kiểm định cho mọi cặp model:
      - paired bootstrap + Wilcoxon trên CCS trung bình / từng ảnh
      - paired bootstrap trên F1@0.5 (IoU) tổng hợp
      - Spearman giữa xếp hạng theo CCS và xếp hạng theo F1@0.5 trên toàn bộ model
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Statistical Significance Testing")
    print("=" * 70)

    models = list(all_results.keys())
    n_pairs = len(models) * (len(models) - 1) // 2
    alpha_bonferroni = 0.05 / n_pairs if n_pairs > 0 else 0.05
    print(f"\n  Số cặp model: {n_pairs} -> ngưỡng Bonferroni-adjusted alpha = {alpha_bonferroni:.5f}")

    def ccs_mean_fn(imgs):
        return float(np.mean([img["ccs"]["ccs"] for img in imgs]))

    # F1@0.5 (IoU) tổng hợp — cần _match_boxes_for_class_fn từ script chính
    def make_f1_fn(thresh, metric):
        def f1_fn(imgs):
            tp = fp = fn = 0
            for img in imgs:
                ai_boxes, gt_boxes = img["ai_boxes"], img["gt_boxes"]
                classes = set(ai_boxes.keys()) | set(gt_boxes.keys())
                for cls_id in classes:
                    t, f, n = match_boxes_for_class_fn(ai_boxes.get(cls_id, []), gt_boxes.get(cls_id, []), thresh, metric)
                    tp += t; fp += f; fn += n
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1_fn

    f1_05_fn = make_f1_fn(0.5, "iou")

    pairwise_results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mA, mB = models[i], models[j]
            imgs_a = all_results[mA]["per_image"]
            imgs_b = all_results[mB]["per_image"]

            ccs_a = [img["ccs"]["ccs"] for img in imgs_a]
            ccs_b = [img["ccs"]["ccs"] for img in imgs_b]

            boot_ccs = paired_bootstrap_diff(imgs_a, imgs_b, ccs_mean_fn, n_boot=n_boot_ccs)
            wilcox_ccs = wilcoxon_signed_rank(ccs_a, ccs_b)
            boot_f1 = paired_bootstrap_diff(imgs_a, imgs_b, f1_05_fn, n_boot=n_boot_f1)

            result = {
                "model_A": mA,
                "model_B": mB,
                "ccs_bootstrap": boot_ccs,
                "ccs_wilcoxon": wilcox_ccs,
                "f1_0.5_bootstrap": boot_f1,
                "significant_ccs_bonferroni": bool(boot_ccs["p_value"] < alpha_bonferroni),
                "significant_f1_bonferroni": bool(boot_f1["p_value"] < alpha_bonferroni),
            }
            pairwise_results.append(result)

            print(f"\n  [{mA} vs {mB}]")
            print(f"    CCS: diff={boot_ccs['observed_diff']:+.4f}  95%CI=[{boot_ccs['ci95_low']:+.4f}, {boot_ccs['ci95_high']:+.4f}]  "
                  f"p={boot_ccs['p_value']:.4f}  Wilcoxon p={wilcox_ccs.get('p_value', 'N/A')}")
            print(f"    F1@0.5(IoU): diff={boot_f1['observed_diff']:+.4f}  95%CI=[{boot_f1['ci95_low']:+.4f}, {boot_f1['ci95_high']:+.4f}]  "
                  f"p={boot_f1['p_value']:.4f}")
            if boot_ccs["p_value"] >= alpha_bonferroni:
                print(f"    -> KHÔNG có ý nghĩa thống kê sau hiệu chỉnh Bonferroni (p >= {alpha_bonferroni:.5f})")

    # Spearman giữa xếp hạng CCS và xếp hạng F1@0.5 trên toàn bộ model
    ccs_means = [ccs_mean_fn(all_results[m]["per_image"]) for m in models]
    f1_means = [f1_05_fn(all_results[m]["per_image"]) for m in models]
    rho = spearman_rank_correlation(ccs_means, f1_means)

    print(f"\n  Spearman rho(xếp hạng CCS, xếp hạng F1@0.5 IoU) trên {len(models)} model = {rho:.4f}")
    print("  (rho gần 1 nghĩa là 2 cách xếp hạng gần như giống nhau tổng thể — nếu CCS")
    print("   thay đổi thứ hạng đáng kể so với F1, rho sẽ thấp hơn hẳn 1.)")

    output = {
        "n_models": len(models),
        "n_pairs": n_pairs,
        "bonferroni_alpha": round(alpha_bonferroni, 6),
        "spearman_ccs_vs_f1_0.5": round(rho, 4),
        "pairwise": pairwise_results,
    }
    save_path = Path(output_dir) / "significance_tests.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return output
