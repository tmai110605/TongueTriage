"""
alpha_beta_optimization.py — Data-driven alpha/beta weighting for CCS.

VẤN ĐỀ: CCS = alpha*C_sp + beta*C_sem với alpha=beta=0.5 mặc định, tự nhận
trong CSS.md là "chưa tối ưu". Vì bài 1 KHÔNG có nhãn đồng thuận bác sĩ
(việc đó thuộc bài 2 — acceptance study), ta KHÔNG THỂ tối ưu alpha/beta để
khớp với "độ chấp nhận lâm sàng" — làm vậy sẽ lẫn scope 2 bài.

Thay vào đó, module này dùng các phương pháp TRỌNG SỐ KHÁCH QUAN
(objective weighting) chuẩn trong multi-criteria decision analysis (MCDM),
chỉ dựa trên ĐỘ BIẾN THIÊN/PHÂN BIỆT của chính C_sp và C_sem quan sát được
trên dữ liệu — không cần nhãn ngoài:

  1. Entropy weight method (Shannon entropy) — tiêu chí nào biến thiên/phân
     biệt được các mẫu nhiều hơn thì được trọng số cao hơn. Kỹ thuật kinh
     điển trong MCDM (bắt nguồn từ Shannon 1948, áp dụng rộng rãi từ thập
     niên 1980s).
  2. Std-proportional weighting (~ CRITIC method rút gọn cho đúng 2 tiêu
     chí — với 2 tiêu chí, số hạng tương quan trong CRITIC tự triệt tiêu,
     chỉ còn lại tỉ lệ độ lệch chuẩn).
  3. Ranking-stability analysis — quét alpha 0->1, xem thứ hạng 6 model có
     đổi hay không, tìm khoảng alpha "an toàn" quanh 0.5 và các điểm alpha
     mà thứ hạng thực sự đổi (transition points).

LƯU Ý QUAN TRỌNG PHẢI GHI RÕ TRONG BÀI: các trọng số này phản ánh "tiêu chí
nào phân biệt tốt hơn về mặt THỐNG KÊ giữa các mẫu/model quan sát được",
KHÔNG phản ánh "bác sĩ coi trọng không gian hay ngữ nghĩa hơn trong thực
tế lâm sàng". Đó là 2 câu hỏi khác nhau; câu hỏi sau cần acceptance study.
"""

import math
import json
from pathlib import Path

import numpy as np


# ─── Objective weighting methods ────────────────────────────────────────────

def entropy_weights(csp_values, csem_values):
    """
    Shannon entropy weighting. Trả về (alpha, beta), alpha+beta=1.
    csp_values, csem_values: nên là TOÀN BỘ giá trị per-image, gộp từ mọi
    model (càng nhiều mẫu, ước lượng entropy càng ổn định).
    """
    csp = np.asarray(csp_values, dtype=np.float64)
    csem = np.asarray(csem_values, dtype=np.float64)
    eps = 1e-12
    n = len(csp)
    if n < 2:
        return 0.5, 0.5

    def _entropy(x):
        x = np.clip(x, eps, None)
        p = x / x.sum()
        k = 1.0 / math.log(n)
        return -k * np.sum(p * np.log(p))

    e_sp, e_sem = _entropy(csp), _entropy(csem)
    d_sp, d_sem = 1 - e_sp, 1 - e_sem
    if (d_sp + d_sem) <= 0:
        return 0.5, 0.5
    alpha = d_sp / (d_sp + d_sem)
    return float(alpha), float(1 - alpha)


def std_proportional_weights(csp_values, csem_values):
    """
    Trọng số tỉ lệ độ lệch chuẩn (~ CRITIC rút gọn cho 2 tiêu chí).
    Tiêu chí biến thiên nhiều hơn giữa các mẫu được trọng số cao hơn.
    """
    csp = np.asarray(csp_values, dtype=np.float64)
    csem = np.asarray(csem_values, dtype=np.float64)
    s_sp, s_sem = csp.std(ddof=1), csem.std(ddof=1)
    if (s_sp + s_sem) <= 0:
        return 0.5, 0.5
    alpha = s_sp / (s_sp + s_sem)
    return float(alpha), float(1 - alpha)


# ─── Ranking stability analysis ─────────────────────────────────────────────

def ranking_stability_analysis(all_results, alphas=None):
    """
    Với mỗi alpha, xếp hạng các model theo CCS=alpha*C_sp+(1-alpha)*C_sem
    (dùng mean C_sp/C_sem của từng model). Trả về:
      - rankings_by_alpha: xếp hạng ứng với mỗi alpha
      - transition_points: các alpha mà thứ hạng đổi so với alpha liền trước
      - stable_range_around_0.5: khoảng alpha rộng nhất chứa 0.5 mà thứ hạng
        không đổi so với alpha=0.5
    """
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.001, 0.05)]

    model_means = {}
    for model_name, data in all_results.items():
        per_image = data["per_image"]
        c_sp = float(np.mean([img["ccs"]["c_sp"] for img in per_image]))
        c_sem = float(np.mean([img["ccs"]["c_sem"] for img in per_image]))
        model_means[model_name] = (c_sp, c_sem)

    rankings = {}
    for a in alphas:
        b = 1 - a
        scores = {m: a * cs + b * ce for m, (cs, ce) in model_means.items()}
        ranked = [m for m, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        rankings[a] = ranked

    transition_points = []
    prev = None
    for a in alphas:
        if prev is not None and rankings[a] != prev:
            transition_points.append(a)
        prev = rankings[a]

    closest_to_half = min(alphas, key=lambda x: abs(x - 0.5))
    base_ranking = rankings[closest_to_half]
    stable_alphas = [a for a in alphas if rankings[a] == base_ranking]
    stable_lo, stable_hi = (min(stable_alphas), max(stable_alphas)) if stable_alphas else (None, None)

    return {
        "rankings_by_alpha": {f"{a:.2f}": r for a, r in rankings.items()},
        "transition_points": transition_points,
        "stable_range_around_0.5": [stable_lo, stable_hi],
        "ranking_at_0.5": base_ranking,
    }


# ─── Experiment wrapper ─────────────────────────────────────────────────────

def exp_alpha_beta_datadriven(all_results, output_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Data-Driven Alpha/Beta Weighting (thay cho mặc định 0.5/0.5)")
    print("=" * 70)

    # gộp per-image C_sp, C_sem từ MỌI model để ước lượng trọng số khách quan
    all_csp, all_csem = [], []
    for data in all_results.values():
        for img in data["per_image"]:
            all_csp.append(img["ccs"]["c_sp"])
            all_csem.append(img["ccs"]["c_sem"])

    alpha_ent, beta_ent = entropy_weights(all_csp, all_csem)
    alpha_std, beta_std = std_proportional_weights(all_csp, all_csem)

    print(f"\n  Số mẫu per-image gộp từ {len(all_results)} model: {len(all_csp)}")
    print(f"  Entropy weight       : alpha={alpha_ent:.4f}  beta={beta_ent:.4f}")
    print(f"  Std-proportional (~CRITIC): alpha={alpha_std:.4f}  beta={beta_std:.4f}")
    print(f"  Mặc định hiện dùng    : alpha=0.5000  beta=0.5000")

    stability = ranking_stability_analysis(all_results)
    print(f"\n  Xếp hạng model tại alpha=0.5: {stability['ranking_at_0.5']}")
    print(f"  Khoảng alpha ổn định quanh 0.5 (thứ hạng không đổi): "
          f"[{stability['stable_range_around_0.5'][0]}, {stability['stable_range_around_0.5'][1]}]")
    if stability["transition_points"]:
        print(f"  Các mốc alpha mà thứ hạng đổi: {stability['transition_points']}")
    else:
        print("  Thứ hạng KHÔNG đổi trên toàn miền alpha 0->1 -> kết luận về model tốt nhất")
        print("  không phụ thuộc cách chọn alpha/beta (kết quả rất mạnh nếu đúng).")

    # xếp hạng ứng với alpha khách quan (entropy) để so sánh với alpha=0.5
    a_ent_rounded = round(alpha_ent, 2)
    closest_alpha = min(stability["rankings_by_alpha"].keys(), key=lambda k: abs(float(k) - a_ent_rounded))
    ranking_at_entropy_alpha = stability["rankings_by_alpha"][closest_alpha]
    same_as_default = ranking_at_entropy_alpha == stability["ranking_at_0.5"]
    print(f"\n  Xếp hạng tại alpha=entropy({alpha_ent:.2f}): {ranking_at_entropy_alpha}")
    print(f"  Có giống xếp hạng tại alpha=0.5 không? {'CÓ' if same_as_default else 'KHÔNG'}")

    result = {
        "n_pooled_samples": len(all_csp),
        "entropy_weight": {"alpha": round(alpha_ent, 4), "beta": round(beta_ent, 4)},
        "std_proportional_weight": {"alpha": round(alpha_std, 4), "beta": round(beta_std, 4)},
        "default_weight": {"alpha": 0.5, "beta": 0.5},
        "ranking_stability": stability,
        "ranking_matches_default_at_entropy_alpha": same_as_default,
    }
    save_path = Path(output_dir) / "alpha_beta_datadriven.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {save_path}")
    return result
