"""
Continuous Concordance Score (CCS) for TCM Tongue Diagnosis.

Implements the full CCS pipeline as described in the research proposal:
  C_sp  – Spatial concordance with Gaussian Distance Decay
  C_sem – Semantic concordance with Taxonomy + Wu-Palmer similarity
    CCS   = α·C_sp + β·C_sem
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─── Taxonomy & Wu-Palmer ──────────────────────────────────────────────────

TAXONOMY_TREE = {
    "name": "Tongue symptom",
    "children": [
        {
            # Both hongshe (uniformly red tongue body) and hongdianshe
            # (red-spotted tongue body) describe the same underlying
            # feature — abnormal redness of the tongue body — differing
            # only in distribution (uniform vs. patchy). Grouping them
            # under one "Color" node gives them a non-trivial Wu-Palmer
            # similarity (LCS = Color, not the taxonomy root), whereas the
            # previous tree put hongdianshe under "Shape" alongside
            # pangdashe/liewenshe/chihenshe, which made
            # wu_palmer(hongshe, hongdianshe) == 0 — the same score as
            # comparing hongshe to a teeth-marked tongue, clinically an
            # unrelated feature.
            #
            # PROVISIONAL: this taxonomy is an experimental structure
            # built to make C_sem computable for these 8 classes; it has
            # not been validated by a TCM domain expert. Treat groupings
            # as a documented, revisable assumption in the paper (e.g.
            # "provisional taxonomy, Appendix X"), not as an established
            # clinical ontology.
            "name": "Color",
            "children": [
                {"name": "hongshe", "id": 1},
                {"name": "hongdianshe", "id": 3},
            ],
        },
        {
            "name": "Coating",
            "children": [
                {
                    "name": "Coating nature",
                    "children": [
                        {"name": "botaishe", "id": 0},
                    ],
                },
                {
                    "name": "Coating color",
                    "children": [
                        {"name": "baitaishe", "id": 6},
                        {"name": "huangtaishe", "id": 7},
                    ],
                },
            ],
        },
        {
            "name": "Shape",
            "children": [
                {"name": "pangdashe", "id": 2},
                {"name": "liewenshe", "id": 4},
                {"name": "chihenshe", "id": 5},
            ],
        },
    ],
}


def _flatten(node, parent=None, depth=0, path=None):
    """Return {name: (depth, parent_name, full_path)} for every leaf & internal node."""
    if path is None:
        path = []
    result = {}
    current_path = path + [node["name"]]
    info = {
        "depth": depth,
        "parent": parent,
        "path": current_path,
        "children": node.get("children", []),
    }
    if "id" in node:  # leaf (symptom)
        info["id"] = node["id"]
    result[node["name"]] = info
    for child in node.get("children", []):
        result.update(_flatten(child, node["name"], depth + 1, current_path))
    return result


_NODE_INFO = _flatten(TAXONOMY_TREE)

# Single source of truth for the id <-> class-name mapping: derived directly
# from TAXONOMY_TREE instead of being maintained as a second, separate list.
# compute_c_sem/build_semantic_matrix assume `sem_matrix[k]` (row k) refers to
# class_id k; this construction guarantees that invariant holds even if the
# taxonomy is edited later, rather than relying on someone keeping a manual
# CLASS_NAMES_8 list in sync with the ids inside TAXONOMY_TREE by hand.
CLASS_ID_TO_NAME = {
    info["id"]: name for name, info in _NODE_INFO.items() if "id" in info
}
_missing = [i for i in range(len(CLASS_ID_TO_NAME)) if i not in CLASS_ID_TO_NAME]
assert not _missing, f"TAXONOMY_TREE is missing class id(s): {_missing}"
CLASS_NAMES_8 = [CLASS_ID_TO_NAME[i] for i in range(len(CLASS_ID_TO_NAME))]


def _find_lcs(name_a, name_b):
    """Find the Lowest Common Subsumer of two node names."""
    info_a = _NODE_INFO.get(name_a)
    info_b = _NODE_INFO.get(name_b)
    if info_a is None or info_b is None:
        return None
    path_a = info_a["path"]
    path_b = info_b["path"]
    lcs = None
    for pa, pb in zip(path_a, path_b):
        if pa == pb:
            lcs = pa
        else:
            break
    return lcs


def wu_palmer_similarity(class_a, class_b):
    """Wu-Palmer similarity between two class names in [0, 1]."""
    if class_a == class_b:
        return 1.0
    lcs_name = _find_lcs(class_a, class_b)
    if lcs_name is None:
        return 0.0
    lcs_info = _NODE_INFO[lcs_name]
    info_a = _NODE_INFO[class_a]
    info_b = _NODE_INFO[class_b]
    d_root_lcs = lcs_info["depth"]
    d_a_lcs = info_a["depth"] - lcs_info["depth"]
    d_b_lcs = info_b["depth"] - lcs_info["depth"]
    if 2 * d_root_lcs + d_a_lcs + d_b_lcs == 0:
        return 0.0
    return (2.0 * d_root_lcs) / (2.0 * d_root_lcs + d_a_lcs + d_b_lcs)


def build_semantic_matrix(class_names=None):
    """Build W_sem ∈ [0,1]^(N×N) where w_ij = WuPalmer(i, j)."""
    if class_names is None:
        class_names = CLASS_NAMES_8
    n = len(class_names)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = wu_palmer_similarity(class_names[i], class_names[j])
    return mat


def print_semantic_matrix(class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES_8
    mat = build_semantic_matrix(class_names)
    print("Semantic similarity matrix (Wu-Palmer):")
    header = f"{'':<14}" + "".join(f"{c:<14}" for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<14}" + "".join(f"{mat[i, j]:<14.4f}" for j in range(len(class_names)))
        print(row)
    return mat


# ─── Spatial Concordance (Gaussian Distance Decay) ─────────────────────────

def _box_to_gaussian(box):
    """Convert a bounding box to Gaussian parameters.

    box: (x1, y1, x2, y2) in pixel coordinates.
    Returns: (mu_x, mu_y, sigma_x, sigma_y)

    NOTE: This is scale-invariant by construction (all downstream ratios
    cancel out any uniform scaling), so it does NOT need img_size — as long
    as box_ai and box_gt are expressed in the SAME pixel coordinate system
    (i.e. both refer to the same original image). Getting that consistency
    right is the caller's responsibility; see parse_yolo_labels /
    evaluate_model_on_dataset for where the actual image size matters.
    """
    x1, y1, x2, y2 = box
    mu_x = (x1 + x2) / 2.0
    mu_y = (y1 + y2) / 2.0
    sigma_x = max((x2 - x1) / 4.0, 1e-6)
    sigma_y = max((y2 - y1) / 4.0, 1e-6)
    return mu_x, mu_y, sigma_x, sigma_y


def spatial_concordance_for_pair(box_ai, box_gt):
    """Compute s_k for a single (AI, ground-truth) box pair.

    Formula (closed-form Gaussian overlap):
      s_k = sqrt(2σx_ai·σx_gt / (σx_ai² + σx_gt²))
          · sqrt(2σy_ai·σy_gt / (σy_ai² + σy_gt²))
          · exp( -Δx² / (2·(σx_ai² + σx_gt²)) - Δy² / (2·(σy_ai² + σy_gt²)) )

    NAMING (methods-section note): this is the normalized L2 inner product
    (cosine similarity in function space) between the two Gaussian density
    functions G_ai and G_gt — i.e. s_k = <G_ai, G_gt> / (||G_ai||·||G_gt||)
    — NOT the Bhattacharyya coefficient (which is defined via ∫sqrt(G_ai·G_gt)
    and has a *different* closed form with 4·(σ_ai²+σ_gt²) in the exponent
    denominator). Deriving the L2 cosine similarity of two 1D Gaussians
    analytically gives exactly the exponent denominator of 2 used here, so
    the formula itself is correct — call it "normalized Gaussian overlap" or
    "Gaussian cosine similarity" in the paper, not "Bhattacharyya
    coefficient", since that name refers to the other (differently-scaled)
    quantity.
    """
    mx_ai, my_ai, sx_ai, sy_ai = _box_to_gaussian(box_ai)
    mx_gt, my_gt, sx_gt, sy_gt = _box_to_gaussian(box_gt)

    dx = mx_ai - mx_gt
    dy = my_ai - my_gt
    sx_ai2 = sx_ai * sx_ai
    sy_ai2 = sy_ai * sy_ai
    sx_gt2 = sx_gt * sx_gt
    sy_gt2 = sy_gt * sy_gt

    scale_x = math.sqrt(2.0 * sx_ai * sx_gt / (sx_ai2 + sx_gt2))
    scale_y = math.sqrt(2.0 * sy_ai * sy_gt / (sy_ai2 + sy_gt2))
    spatial_exp = math.exp(
        -(dx * dx) / (2.0 * (sx_ai2 + sx_gt2))
        - (dy * dy) / (2.0 * (sy_ai2 + sy_gt2))
    )
    return scale_x * scale_y * spatial_exp


def compute_c_sp(ai_boxes, gt_boxes):
    """Spatial concordance C_sp across all labels.

    ai_boxes: dict {class_id: [(x1,y1,x2,y2,conf), ...]}
    gt_boxes: dict {class_id: [(x1,y1,x2,y2), ...]}

    Both must already be in the SAME pixel coordinate system (i.e. the
    actual dimensions of the image they came from). This function no
    longer takes img_size because it never needed it — see
    parse_yolo_labels / evaluate_model_on_dataset for where getting the
    real per-image size right actually matters.

    Returns:
      C_sp  – float in [0, 1]
      per_class – dict {class_id: s_k} for debugging
    """
    L_ai = set(ai_boxes.keys())
    L_gt = set(gt_boxes.keys())
    L = L_ai | L_gt

    if not L:
        return 1.0, {}

    s_vals = {}
    for k in L:
        ai_list = ai_boxes.get(k, [])
        gt_list = gt_boxes.get(k, [])

        if not ai_list or not gt_list:
            # False positive or false negative → penalty
            s_vals[k] = 0.0
            continue

        # DESIGN ASSUMPTION: this scores each class k as a single
        # region-level concordance value, not per-instance detection. If
        # ai_list/gt_list contain more than one box for the same class, we
        # take the single best-matching (AI, GT) pair and ignore the rest.
        # This is appropriate for tongue symptoms that are annotated as at
        # most one region per class per image (the TMC-Tongue 8-class setup
        # this pipeline targets). It does NOT penalize duplicate/spurious
        # extra boxes within a class (e.g. AI predicting the same symptom
        # twice) — if the dataset or model can produce multiple genuine
        # instances of one class per image, this function under-counts
        # false positives and would need one-to-one instance matching
        # (e.g. Hungarian matching) instead.
        best_s = 0.0
        for a_box in ai_list:
            a_box_xy = a_box[:4]
            for g_box in gt_list:
                s = spatial_concordance_for_pair(a_box_xy, g_box)
                if s > best_s:
                    best_s = s
        s_vals[k] = best_s

    c_sp = sum(s_vals.values()) / len(L)
    return c_sp, s_vals


# ─── Semantic Concordance ──────────────────────────────────────────────────

def compute_c_sem(ai_labels, gt_labels, sem_matrix):
    """Semantic concordance C_sem.

    ai_labels: set of class_ids detected by AI.
    gt_labels: set of class_ids annotated by doctor.
    sem_matrix: N×N Wu-Palmer similarity matrix (sem_matrix[i, j] = row i,
      column j — indices must correspond to class_id, see CLASS_ID_TO_NAME).

    For each label k in the union of ai_labels and gt_labels:
      - k in both sets            -> perfect match (sem_matrix[k, k] == 1.0)
      - k in ai_labels only (FP)  -> best semantic match against gt_labels
      - k in gt_labels only (FN)  -> best semantic match against ai_labels
    This treats over-detection (FP) and missed findings (FN) symmetrically:
    both get partial credit when the *other* side has something semantically
    close, and both get 0 only when the other side is completely empty. This
    mirrors how compute_c_sp treats FP/FN symmetrically (both -> s_k = 0 when
    there is no box on the other side to compare against).

    Returns:
      C_sem – float in [0, 1]
    """
    L = ai_labels | gt_labels
    if not L:
        return 1.0

    total = 0.0
    for k in L:
        if k in ai_labels and k in gt_labels:
            total += 1.0  # exact match; sem_matrix[k, k] is defined as 1.0
        elif k in ai_labels:
            # False positive: AI predicted k but the doctor didn't annotate
            # it here -> credit how close k is to the nearest true label.
            total += max(sem_matrix[k, j] for j in gt_labels) if gt_labels else 0.0
        else:
            # False negative: doctor annotated k but AI missed it -> credit
            # how close k is to the nearest label the AI *did* predict.
            total += max(sem_matrix[k, j] for j in ai_labels) if ai_labels else 0.0

    # Cast away from numpy.float32 (sem_matrix's dtype) to a plain Python
    # float: numpy.float32 fails `isinstance(v, float)` checks (see demo())
    # and isn't JSON-serializable by default.
    return float(total / len(L))


# ─── CCS (combined score) ───────────────────────────────────────────────────

def compute_ccs(ai_boxes, gt_boxes, sem_matrix, alpha=0.5, beta=0.5):
    """Full Continuous Concordance Score.

    ai_boxes / gt_boxes must already be in the same pixel coordinate
    system (see compute_c_sp).

    CCS = α·C_sp + β·C_sem

    Returns dict with all components for analysis.
    """
    ai_labels = set(ai_boxes.keys())
    gt_labels = set(gt_boxes.keys())

    # 1. Spatial
    c_sp, sp_per_class = compute_c_sp(ai_boxes, gt_boxes)

    # 2. Semantic
    c_sem = compute_c_sem(ai_labels, gt_labels, sem_matrix)

    # 3. CCS
    ccs = alpha * c_sp + beta * c_sem

    return {
        "ccs": ccs,
        "c_sp": c_sp,
        "c_sem": c_sem,
        "sp_per_class": sp_per_class,
        "alpha": alpha,
        "beta": beta,
    }


# ─── Inference wrapper ──────────────────────────────────────────────────────

def parse_yolo_preds(results, conf_thresh=0.25):
    """Parse YOLO prediction results into ai_boxes dict.
    
    results: ultralytics Results object.
    Returns: {cls_id: [(x1,y1,x2,y2,conf), ...]}
    """
    boxes = defaultdict(list)
    if results.boxes is None:
        return dict(boxes)
    for box in results.boxes:
        conf = box.conf.item()
        if conf < conf_thresh:
            continue
        cls_id = int(box.cls.item())
        xyxy = box.xyxy[0].tolist()  # pixel coords
        boxes[cls_id].append(tuple(xyxy) + (conf,))
    return dict(boxes)


def parse_yolo_labels(label_path, img_size):
    """Parse YOLO-format label file into gt_boxes dict.

    img_size: (width, height) of the ACTUAL image this label file
    corresponds to (no default — passing the wrong size here silently
    misaligns GT boxes against AI boxes in compute_c_sp, since AI boxes
    from Ultralytics come back in the real image's pixel dimensions).

    Returns: {cls_id: [(x1,y1,x2,y2), ...]} in pixel coordinates.
    """
    w, h = img_size
    boxes = defaultdict(list)
    if not Path(label_path).exists():
        return dict(boxes)
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            boxes[cls_id].append((x1, y1, x2, y2))
    return dict(boxes)


def evaluate_model_on_dataset(model_path, data_yaml, label_root, sem_matrix,
                              conf_thresh=0.25, img_size=None,
                              max_images=None, split="test"):
    """Run full CCS evaluation on a dataset split.

    split: which split key to read from data_yaml ("train"/"val"/"test").
    Defaults to "test" for the final benchmark, but exposing it lets the
    same function run on "val" during ablations without editing this file.

    img_size: optional (width, height) OVERRIDE applied to every image.
    Leave as None (default, recommended) so each image's real dimensions
    are read from the YOLO inference result itself (preds.orig_shape),
    guaranteeing GT boxes are converted into the same pixel coordinate
    system that the AI boxes are already in. Only pass an explicit
    img_size if you are certain every image in the split shares that
    exact resolution — a bulk override is a correctness risk on any
    dataset with mixed image sizes.

    Returns list of per-image CCS results.
    """
    import yaml
    from ultralytics import YOLO
    model = YOLO(model_path)
    data_cfg = yaml.safe_load(open(data_yaml))
    img_dir = Path(str(data_cfg["path"])) / data_cfg[split]

    img_paths = sorted(img_dir.glob("*"))
    if max_images:
        img_paths = img_paths[:max_images]

    results_list = []
    for img_path in img_paths:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # AI prediction
        preds = model(img_path, imgsz=640, verbose=False)[0]
        ai_boxes = parse_yolo_preds(preds, conf_thresh)

        # Ground truth — use this image's REAL size unless explicitly overridden,
        # so GT boxes land in the exact same coordinate system as ai_boxes above.
        if img_size is not None:
            actual_img_size = img_size
        else:
            orig_h, orig_w = preds.orig_shape  # ultralytics gives (height, width)
            actual_img_size = (orig_w, orig_h)

        label_path = Path(str(label_root)) / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_labels(label_path, actual_img_size)

        # CCS
        ccs_result = compute_ccs(
            ai_boxes, gt_boxes, sem_matrix,
        )
        ccs_result["image"] = img_path.name
        results_list.append(ccs_result)

    return results_list


# ─── Demo / test ────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("CCS – Continuous Concordance Score Demo")
    print("=" * 60)

    # 1. Semantic matrix
    sem_mat = build_semantic_matrix()
    print_semantic_matrix()

    # 2. Example: simulate one image
    print("\n--- Example: single image ---")
    ai_boxes = {
        1: [(500, 300, 600, 400, 0.85)],   # hongshe
        6: [(700, 350, 850, 500, 0.72)],   # baitaishe
    }
    gt_boxes = {
        1: [(510, 310, 590, 390)],           # hongshe (overlaps well)
        7: [(710, 360, 830, 490)],           # huangtaishe (close to baitaishe)
    }
    # gt also has chihenshe that AI missed
    gt_boxes[5] = [(400, 280, 480, 380)]

    result = compute_ccs(ai_boxes, gt_boxes, sem_mat)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")

    # 3. Full dataset evaluation
    print("\n--- Full test set evaluation ---")
    try:
        data_yaml = "shezhen datasets/shezhenv3-8class/shezhenv3-8class.yaml"
        label_root = "shezhen datasets/shezhenv3-8class/test/labels"
        # Only 10 images for demo
        results = evaluate_model_on_dataset(
            "runs/8class/yolov8m-8class/weights/best.pt",
            data_yaml, label_root, sem_mat,
            max_images=10,
        )
        ccs_vals = [r["ccs"] for r in results]
        print(f"  Processed {len(results)} images")
        print(f"  Mean CCS: {np.mean(ccs_vals):.4f} ± {np.std(ccs_vals):.4f}")
        print(f"  Mean C_sp: {np.mean([r['c_sp'] for r in results]):.4f}")
        print(f"  Mean C_sem: {np.mean([r['c_sem'] for r in results]):.4f}")
    except Exception as e:
        print(f"  (Skipping full eval – model not ready yet): {e}")


if __name__ == "__main__":
    demo()