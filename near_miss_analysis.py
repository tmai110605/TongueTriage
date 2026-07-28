"""
near_miss_analysis.py — Định lượng near-miss vs far-miss error cho các lớp khó.

Trả lời câu hỏi ở §Class Imbalance (dòng 762-766 bài báo): trong số FP/FN của
các lớp khó (hongdianshe, chihenshe, botaishe, liewenshe), bao nhiêu % là
near-miss (best semantic match CÙNG nhánh taxonomy, similarity > 0) so với
far-miss (hoàn toàn KHÁC nhánh, similarity = 0)?

Dùng lại đúng cấu trúc dữ liệu per_image như trong nwd_baseline.py
(ai_boxes / gt_boxes: dict[class] -> list[box]).

CÁCH DÙNG: import và gọi analyze_near_miss(all_results) với all_results có
cùng cấu trúc {model_name: {"per_image": [...]}} như exp_nwd_comparison() đã
dùng — không cần train lại hay chạy detector, chỉ phân tích lại kết quả đã có.

LƯU Ý: nếu ai_boxes/gt_boxes của bạn dùng class_id dạng số thay vì tên lớp
(vd. 0,1,2...), cần map ngược sang tên lớp (botaishe, hongshe, ...) trước khi
gọi hàm này, vì W_SEM bên dưới được đánh index bằng tên lớp để khớp đúng
Table tab:semantic-matrix trong bài.
"""

import json
from pathlib import Path
from collections import defaultdict
CLASS_NAMES = {
    0: "botaishe",
    1: "hongshe",
    2: "pangdashe",
    3: "hongdianshe",
    4: "liewenshe",
    5: "chihenshe",
    6: "baitaishe",
    7: "huangtaishe",
}
# Ma trận Wu-Palmer similarity, lấy nguyên từ Table tab:semantic-matrix trong bài
W_SEM = {
    "botaishe":    {"botaishe": 1.000, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.333, "huangtaishe": 0.333},
    "hongshe":     {"botaishe": 0.000, "hongshe": 1.000, "pangdashe": 0.000, "hongdianshe": 0.500, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "pangdashe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 1.000, "hongdianshe": 0.000, "liewenshe": 0.500, "chihenshe": 0.500, "baitaishe": 0.000, "huangtaishe": 0.000},
    "hongdianshe": {"botaishe": 0.000, "hongshe": 0.500, "pangdashe": 0.000, "hongdianshe": 1.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "liewenshe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 0.500, "hongdianshe": 0.000, "liewenshe": 1.000, "chihenshe": 0.500, "baitaishe": 0.000, "huangtaishe": 0.000},
    "chihenshe":   {"botaishe": 0.000, "hongshe": 0.000, "pangdashe": 0.500, "hongdianshe": 0.000, "liewenshe": 0.500, "chihenshe": 1.000, "baitaishe": 0.000, "huangtaishe": 0.000},
    "baitaishe":   {"botaishe": 0.333, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 1.000, "huangtaishe": 0.667},
    "huangtaishe": {"botaishe": 0.333, "hongshe": 0.000, "pangdashe": 0.000, "hongdianshe": 0.000, "liewenshe": 0.000, "chihenshe": 0.000, "baitaishe": 0.667, "huangtaishe": 1.000},
}

# 4 lớp khó nhất theo Fig perclass-ap: mean AP@[.5:.95] < 0.13
HARD_CLASSES = {"botaishe", "hongdianshe", "liewenshe", "chihenshe"}


def _best_semantic_match(cls_k, other_side_classes):
    """Similarity lớn nhất của cls_k so với các nhãn thực có ở phía đối diện
    (giống hệt quy tắc c_k trong Eq. csem-percls của bài, chỉ khác là ở đây
    ta giữ lại nhãn nào cho ra best-match, để audit thủ công nếu cần)."""
    if not other_side_classes:
        return 0.0, None
    best_sim, best_cls = 0.0, None
    for j in other_side_classes:
        sim = W_SEM.get(cls_k, {}).get(j, 0.0)
        if sim > best_sim:
            best_sim, best_cls = sim, j
    return best_sim, best_cls


def analyze_near_miss(all_results, hard_classes=HARD_CLASSES):

    print("\n" + "=" * 70)
    print("NEAR-MISS vs FAR-MISS ANALYSIS — Hard Classes")
    print("=" * 70)

    summary_rows = []

    # ==========================================================
    # LOOP OVER MODELS
    # ==========================================================
    for model_name, data in all_results.items():

        per_image = data["per_image"]

        # Records của RIÊNG model hiện tại
        records = []

        # ======================================================
        # LOOP OVER IMAGES
        # ======================================================
        for img in per_image:

            ai_boxes = img["ai_boxes"]
            gt_boxes = img["gt_boxes"]

            # --------------------------------------------------
            # Convert class IDs → class names
            # --------------------------------------------------
            ai_classes = {
                CLASS_NAMES[class_id]
                for class_id in ai_boxes.keys()
                if class_id in CLASS_NAMES
            }

            gt_classes = {
                CLASS_NAMES[class_id]
                for class_id in gt_boxes.keys()
                if class_id in CLASS_NAMES
            }

            # --------------------------------------------------
            # False Positive
            # AI predicts hard class, GT does not contain it
            # --------------------------------------------------
            for k in (ai_classes - gt_classes) & hard_classes:

                sim, match = _best_semantic_match(
                    k,
                    gt_classes
                )

                records.append({
                    "image": img.get("img_name", "?"),
                    "class": k,
                    "type": "FP",
                    "best_sim": sim,
                    "matched_with": match,
                })

            # --------------------------------------------------
            # False Negative
            # GT contains hard class, AI does not predict it
            # --------------------------------------------------
            for k in (gt_classes - ai_classes) & hard_classes:

                sim, match = _best_semantic_match(
                    k,
                    ai_classes
                )

                records.append({
                    "image": img.get("img_name", "?"),
                    "class": k,
                    "type": "FN",
                    "best_sim": sim,
                    "matched_with": match,
                })

        # ======================================================
        # AGGREGATE RESULTS FOR THIS MODEL
        # ======================================================

        n_total = len(records)

        n_near = sum(
            1
            for r in records
            if r["best_sim"] > 0
        )

        n_far = n_total - n_near

        pct_near = (
            round(100 * n_near / n_total, 1)
            if n_total
            else 0.0
        )

        # ------------------------------------------------------
        # Per-class statistics
        # ------------------------------------------------------

        per_class = defaultdict(
            lambda: {
                "total": 0,
                "near": 0
            }
        )

        for r in records:

            per_class[r["class"]]["total"] += 1

            if r["best_sim"] > 0:
                per_class[r["class"]]["near"] += 1

        per_class_pct = {

            k: {
                "total": v["total"],
                "near": v["near"],
                "pct_near": round(
                    100 * v["near"] / v["total"],
                    1
                )
                if v["total"]
                else 0.0,
            }

            for k, v in per_class.items()
        }

        # ======================================================
        # ONE SUMMARY ROW PER MODEL
        # ======================================================

        summary_rows.append({

            "model": model_name,

            "total_hardclass_errors": n_total,

            "near_miss": n_near,

            "far_miss": n_far,

            "pct_near_miss": pct_near,

            "per_class": per_class_pct,

        })

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    header = (
        f"{'Model':<10} "
        f"{'Total':<8} "
        f"{'Near-miss':<10} "
        f"{'Far-miss':<10} "
        f"{'% Near':<8}"
    )

    print(f"\n{header}")
    print("-" * len(header))

    for r in summary_rows:

        print(
            f"{r['model']:<10} "
            f"{r['total_hardclass_errors']:<8} "
            f"{r['near_miss']:<10} "
            f"{r['far_miss']:<10} "
            f"{r['pct_near_miss']:<8}"
        )

    # ==========================================================
    # POOLED RESULT ACROSS ALL MODELS
    # ==========================================================

    total_all = sum(
        r["total_hardclass_errors"]
        for r in summary_rows
    )

    near_all = sum(
        r["near_miss"]
        for r in summary_rows
    )

    pooled_pct = (
        round(100 * near_all / total_all, 1)
        if total_all
        else 0.0
    )

    print(
        f"\n  Pooled across all models: "
        f"{near_all}/{total_all} "
        f"({pooled_pct}%) "
        f"of hard-class errors are near-miss confusions."
    )

    return summary_rows, pooled_pct
    """
    Với mỗi model: gom toàn bộ FP/FN của các lớp khó, phân loại
    near-miss (best_sim > 0, tức LCS cùng nhánh taxonomy) vs
    far-miss (best_sim == 0, khác nhánh hoàn toàn / không có nhãn nào ở
    phía đối diện). Đây chính là số liệu định lượng cho tuyên bố ở
    §Class Imbalance của bài (hiện đang chỉ suy luận định tính).
    """
    print("\n" + "=" * 70)
    print("NEAR-MISS vs FAR-MISS ANALYSIS — Hard Classes")
    print("=" * 70)

    summary_rows = []
    for model_name, data in all_results.items():
        per_image = data["per_image"]
        records = []  # chi tiết từng FP/FN, để soi lại ảnh cụ thể nếu cần

    for img in per_image:

        ai_boxes = img["ai_boxes"]
        gt_boxes = img["gt_boxes"]

        # ==========================================================
        # Convert class IDs → class names
        # ==========================================================
        ai_classes = {
            CLASS_NAMES[class_id]
            for class_id in ai_boxes.keys()
            if class_id in CLASS_NAMES
        }

        gt_classes = {
            CLASS_NAMES[class_id]
            for class_id in gt_boxes.keys()
            if class_id in CLASS_NAMES
        }

        # ==========================================================
        # False Positive:
        # AI predicts class, GT does not contain class
        # ==========================================================
        for k in (ai_classes - gt_classes) & hard_classes:

            sim, match = _best_semantic_match(
                k,
                gt_classes
            )

            records.append({
                "image": img.get("img_name", "?"),
                "class": k,
                "type": "FP",
                "best_sim": sim,
                "matched_with": match,
            })

        # ==========================================================
        # False Negative:
        # GT contains class, AI does not predict class
        # ==========================================================
        for k in (gt_classes - ai_classes) & hard_classes:

            sim, match = _best_semantic_match(
                k,
                ai_classes
            )

            records.append({
                "image": img.get("img_name", "?"),
                "class": k,
                "type": "FN",
                "best_sim": sim,
                "matched_with": match,
            })
        n_total = len(records)
        n_near = sum(1 for r in records if r["best_sim"] > 0)
        n_far = n_total - n_near
        pct_near = round(100 * n_near / n_total, 1) if n_total else 0.0

        # tách theo từng lớp khó để biết lớp nào "được cứu" nhiều nhất
        per_class = defaultdict(lambda: {"total": 0, "near": 0})
        for r in records:
            per_class[r["class"]]["total"] += 1
            if r["best_sim"] > 0:
                per_class[r["class"]]["near"] += 1
        per_class_pct = {
            k: {"total": v["total"], "near": v["near"],
                "pct_near": round(100 * v["near"] / v["total"], 1) if v["total"] else 0.0}
            for k, v in per_class.items()
        }

        summary_rows.append({
            "model": model_name,
            "total_hardclass_errors": n_total,
            "near_miss": n_near,
            "far_miss": n_far,
            "pct_near_miss": pct_near,
            "per_class": per_class_pct,
        })

    header = f"{'Model':<10} {'Total':<8} {'Near-miss':<10} {'Far-miss':<10} {'% Near':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['model']:<10} {r['total_hardclass_errors']:<8} "
              f"{r['near_miss']:<10} {r['far_miss']:<10} {r['pct_near_miss']:<8}")

    # Trung bình trên toàn bộ 6 model — con số dùng để chèn thẳng vào bài
    total_all = sum(r["total_hardclass_errors"] for r in summary_rows)
    near_all = sum(r["near_miss"] for r in summary_rows)
    pooled_pct = round(100 * near_all / total_all, 1) if total_all else 0.0
    print(f"\n  Pooled across all models: {near_all}/{total_all} "
          f"({pooled_pct}%) of hard-class errors are near-miss confusions.")

    return summary_rows, pooled_pct


def save_near_miss_report(summary_rows, pooled_pct, output_dir):
    save_path = Path(output_dir) / "near_miss_analysis.json"
    with open(save_path, "w") as f:
        json.dump({"pooled_pct_near_miss": pooled_pct, "rows": summary_rows},
                   f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {save_path}")
