"""
instance_exposure_stats.py — Quantify the "blind spots" of class-level scoring
(Section: Design Choices and Assumptions / A(12).jpg qualitative example).

Supports COCO (.json) and YOLO (.yaml) formats.

Idea: Since C_sp/C_sem only keeps exactly 1 best AI-GT pair per class/image
(Section: Design Choices), any GT instance beyond 1 in the same image/class
will not be scored (neither rewarded nor penalized) whether the model detects it or not.
This script counts, for each class: average/max instances per
image, and what % of total GT boxes for that class falls into this "blind spot".

USAGE:
    # For YOLO format (current project)
    python instance_exposure_stats.py "shezhen datasets/shezhenv3-8class/shezhenv3-8class.yaml" val
"""

import json
import os
import glob
from collections import defaultdict


def instance_exposure_stats(annotation_path, split='val'):
    counts = defaultdict(lambda: defaultdict(int))
    cat_id_to_name = {}

    if annotation_path.endswith('.json'):
        with open(annotation_path, encoding='utf-8') as f:
            coco = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        for ann in coco["annotations"]:
            counts[ann["image_id"]][ann["category_id"]] += 1
            
    elif annotation_path.endswith('.yaml') or annotation_path.endswith('.yml'):
        import yaml
        with open(annotation_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        cat_id_to_name = data.get('names', {})
        if isinstance(cat_id_to_name, list):
            cat_id_to_name = {i: name for i, name in enumerate(cat_id_to_name)}
            
        base_path = data.get('path', os.path.dirname(annotation_path))
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(annotation_path), base_path)
            
        split_path = data.get(split)
        if not split_path:
            raise ValueError(f"Could not find split '{split}' in yaml file.")
        
        # Label directory path instead of images
        if 'images' in split_path:
            labels_path = split_path.replace('images', 'labels')
        else:
            labels_path = 'labels'
            
        full_labels_path = os.path.normpath(os.path.join(base_path, labels_path))
        # Fallback: sometimes base_path might be the parent directory containing the yaml
        if not os.path.exists(full_labels_path):
            full_labels_path = os.path.normpath(os.path.join(base_path, split_path)).replace('images', 'labels')
            
        if not os.path.exists(full_labels_path):
            raise FileNotFoundError(f"Label directory not found at: {full_labels_path}")
            
        label_files = glob.glob(os.path.join(full_labels_path, "*.txt"))
        if not label_files:
            print(f"Warning: No .txt files found in {full_labels_path}")
            
        for label_file in label_files:
            image_id = os.path.basename(label_file)
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cat_id = int(parts[0])
                    counts[image_id][cat_id] += 1
    else:
        raise ValueError("Unsupported file format. Please provide a .json (COCO) or .yaml (YOLO) file.")

    # Group by class: list of "instances/image", only considering images with >=1 instance of that class
    per_class_counts = defaultdict(list)
    for img_id, cat_counts in counts.items():
        for cat_id, n in cat_counts.items():
            per_class_counts[cat_id].append(n)

    header = f"{'Class':<15}{'Images':<10}{'Mean/img':<10}{'Max/img':<10}{'% GT unscored':<15}"
    print(header)
    print("-" * len(header))

    rows = []
    for cat_id, counts_list in sorted(per_class_counts.items()):
        name = cat_id_to_name.get(cat_id, str(cat_id))
        n_images = len(counts_list)
        mean_n = sum(counts_list) / n_images
        max_n = max(counts_list)
        total_gt = sum(counts_list)
        # "excess" boxes beyond the 1 box kept according to class-level convention
        excess = sum(c - 1 for c in counts_list if c > 1)
        pct_excess = 100 * excess / total_gt if total_gt else 0.0

        print(f"{name:<15}{n_images:<10}{mean_n:<10.2f}{max_n:<10}{pct_excess:<15.1f}")
        rows.append({
            "class": name,
            "n_images_with_class": n_images,
            "mean_instances_per_image": round(mean_n, 2),
            "max_instances_per_image": max_n,
            "total_gt_boxes": total_gt,
            "pct_gt_boxes_unscored": round(pct_excess, 1),
        })

    return rows


if __name__ == "__main__":
    import sys
    # Default to project yaml dataset if no args provided
    default_path = r"shezhen datasets\shezhenv3-8class\shezhenv3-8class.yaml"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    print(f"Analyzing split '{split}' of dataset '{path}'")
    instance_exposure_stats(path, split)


