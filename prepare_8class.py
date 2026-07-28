import shutil
import yaml
from pathlib import Path

SELECTED_CLASSES = [1, 2, 4, 6, 7, 8, 9, 10]
SELECTED_NAMES = {
    1: "botaishe",
    2: "hongshe",
    4: "pangdashe",
    6: "hongdianshe",
    7: "liewenshe",
    8: "chihenshe",
    9: "baitaishe",
    10: "huangtaishe",
}

ROOT = Path(r"/home/thaimq/TongueTriage/shezhen datasets/shezhenv3-txt")
OUT = Path(r"/home/thaimq/TongueTriage/shezhen datasets/shezhenv3-8class")

def process_split(split):
    src_img = ROOT / split / "images"
    src_lab = ROOT / split / "labels"
    dst_img = OUT / split / "images"
    dst_lab = OUT / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)

    old2new = {old: new for new, old in enumerate(SELECTED_CLASSES)}

    for lab_file in src_lab.glob("*.txt"):
        with open(lab_file) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            if old_cls in old2new:
                new_cls = old2new[old_cls]
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")

        if new_lines:
            img_stem = lab_file.stem
            src_img_file = src_img / f"{img_stem}.jpg"
            src_img_file2 = src_img / f"{img_stem}.jpeg"

            if src_img_file.exists():
                shutil.copy2(src_img_file, dst_img / src_img_file.name)
            elif src_img_file2.exists():
                shutil.copy2(src_img_file2, dst_img / src_img_file2.name)
            else:
                continue

            with open(dst_lab / lab_file.name, "w") as f:
                f.writelines(new_lines)

    imgs = len(list(dst_img.glob("*")))
    labs = len(list(dst_lab.glob("*.txt")))
    print(f"  {split}: {imgs} images, {labs} labels")
    return imgs, labs


def build_yaml():
    names = {new: SELECTED_NAMES[old] for new, old in enumerate(SELECTED_CLASSES)}
    cfg = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(SELECTED_CLASSES),
        "names": names,
    }
    with open(OUT / "shezhenv3-8class.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\nYAML saved: {OUT / 'shezhenv3-8class.yaml'}")
    print(f"Classes: {list(names.values())}")


def show_distribution():
    from collections import Counter

    print("\n--- Class distribution (8-class) ---")
    for split in ["train", "val", "test"]:
        lab_dir = OUT / split / "labels"
        cnt = Counter()
        for lab in lab_dir.glob("*.txt"):
            with open(lab) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls = int(line.split()[0])
                        cnt[cls] += 1
        print(f"\n{split}:")
        for cid in sorted(cnt.keys()):
            name = SELECTED_NAMES[SELECTED_CLASSES[cid]]
            print(f"  {cid} ({name}): {cnt[cid]}")


if __name__ == "__main__":
    print("Creating 8-class subset...")
    for split in ["train", "val", "test"]:
        process_split(split)

    build_yaml()
    show_distribution()
    print("\nDone. Train with:")
    print(f'  python train_yolo.py single --model yolov8m --name yolov8m-8class --data "{OUT / "shezhenv3-8class.yaml"}" --epochs 100 --imgsz 640')
