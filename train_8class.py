import argparse
import sys
from pathlib import Path

import torch
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "shezhen datasets" / "shezhenv3-8class" / "shezhenv3-8class.yaml"


class Tee:
    """Duplicate writes to both stdout and a log file."""
    def __init__(self, log_path):
        self.file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


MODEL_REGISTRY = {
    "yolov8n":   "yolov8n.pt",
    "yolov8s":   "yolov8s.pt",
    "yolov8m":   "yolov8m.pt",
    "yolov8l":   "yolov8l.pt",
    "yolov8x":   "yolov8x.pt",
    "yolov10n":  "yolov10n.pt",
    "yolov10s":  "yolov10s.pt",
    "yolov10m":  "yolov10m.pt",
    "yolov10l":  "yolov10l.pt",
    "yolov10x":  "yolov10x.pt",
    "yolov11n":  "yolo11n.pt",
    "yolov11s":  "yolo11s.pt",
    "yolov11m":  "yolo11m.pt",
    "yolov11l":  "yolo11l.pt",
    "yolov11x":  "yolo11x.pt",
}


def resolve_checkpoint(model_key_or_path: str) -> str:
    """Map a registry key (e.g. 'yolov10m') to its .pt checkpoint filename.

    If the value isn't a known key (it's already a path like 'yolov8n.pt'
    or 'runs/train/x/weights/best.pt'), it is returned unchanged.
    """
    return MODEL_REGISTRY.get(model_key_or_path.strip(), model_key_or_path)


def train_single(args):
    ckpt = resolve_checkpoint(args.model)
    print(f"[single] model key='{args.model}' -> checkpoint='{ckpt}'")
    model = YOLO(ckpt)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        amp=not args.no_amp,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=not args.no_cos_lr,
        patience=args.patience,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        val=True,
        save=True,
        save_period=args.save_period,
        deterministic=True,
        seed=args.seed,
        verbose=not args.quiet,
    )


def train_all(args):
    models = args.models.split(",")
    for i, model_key in enumerate(models):
        model_key = model_key.strip()
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model: {model_key}, skipping.")
            continue

        ckpt = resolve_checkpoint(model_key)
        exp_name = f"{model_key}-imgsz{args.imgsz}-e{args.epochs}"

        print(f"\n{'='*60}")
        print(f"Training {model_key} ({i+1}/{len(models)}) -> {Path(args.project) / exp_name}")
        print(f"{'='*60}\n")

        model = YOLO(ckpt)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=exp_name,
            exist_ok=args.exist_ok,
            pretrained=True,
            amp=not args.no_amp,
            workers=args.workers,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            cos_lr=not args.no_cos_lr,
            patience=args.patience,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            val=True,
            save=True,
            save_period=max(args.save_period, args.epochs // 5),
            deterministic=True,
            seed=args.seed,
            verbose=not args.quiet,
        )

        # Clean up to free memory
        del model
        torch.cuda.empty_cache()


def test_model(args):
    model = YOLO(args.weights)
    # Force use of test split (ultralytics uses val by default in val())
    # Override split via data yaml by temporarily setting val=test
    import yaml
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()
    with open(data_path) as f:
        cfg = yaml.safe_load(f)
    test_path = cfg.get("test", None)
    if test_path is None:
        print("Error: data YAML has no 'test' field.")
        sys.exit(1)

    # Build a temporary yaml pointing val -> test
    tmp_yaml = data_path.parent / "_test_only.yaml"
    test_cfg = cfg.copy()
    test_cfg["val"] = test_path
    with open(tmp_yaml, "w") as f:
        yaml.dump(test_cfg, f)

    metrics = model.val(
        data=str(tmp_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name or f"test-{Path(args.weights).stem}",
        exist_ok=args.exist_ok,
        workers=args.workers,
        verbose=not args.quiet,
        save_json=True,
        save_hybrid=False,
    )

    tmp_yaml.unlink(missing_ok=True)
    print(f"\nTest results saved to {args.project}/{args.name}")

    # Print mAP summary
    print(f"\n  mAP@0.5:   {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    for i, (c, ap) in enumerate(zip(metrics.ap_class_index, metrics.box.maps)):
        print(f"    class {c}: mAP@0.5:0.95 = {ap:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train/Test YOLO models on 8-class TMC-Tongue dataset"
    )
    sub = parser.add_subparsers(dest="command")

    # Single model
    p_single = sub.add_parser("single", help="Train a single YOLO model")
    p_single.add_argument(
        "--model", type=str, required=True,
        help=f"Model key or .pt path. Keys: {list(MODEL_REGISTRY.keys())}"
    )

    # Multiple models
    p_all = sub.add_parser("all", help="Train multiple YOLO models sequentially")
    p_all.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated model keys, e.g. yolov8m,yolov10m,yolov11m"
    )

    # Test model
    p_test = sub.add_parser("test", help="Evaluate trained model on test split")
    p_test.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained model weights (.pt)"
    )

    # Shared args
    for p in [p_single, p_all]:
        p.add_argument(
            "--data", type=str,
            default=str(DATA_YAML),
            help="Dataset YAML path"
        )
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--device", type=str, default="0")
        p.add_argument("--project", type=str, default="runs/train")
        p.add_argument("--no-amp", action="store_true")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--lr0", type=float, default=0.01)
        p.add_argument("--lrf", type=float, default=0.01)
        p.add_argument("--weight-decay", type=float, default=0.0005)
        p.add_argument("--warmup-epochs", type=float, default=3.0)

        # ==== THÊM MỚI (theo phân tích results.csv: model overfit sau epoch ~40,
        # val loss tăng trở lại trong khi train loss vẫn giảm) ====
        p.add_argument(
            "--no-cos-lr", action="store_true",
            help="Disable cosine LR schedule. Cosine is ON by default (thường hội tụ mượt "
                 "hơn linear decay mặc định cho các run <=100 epoch)."
        )
        p.add_argument(
            "--patience", type=int, default=20,
            help="Số epoch không cải thiện val trước khi early-stop (mặc định 20). "
                 "Đặt --epochs 100 --patience 100 để tắt hẳn early stopping."
        )
        p.add_argument(
            "--hsv-h", type=float, default=0.015,
            help="Augmentation hue jitter (fraction). Giữ default Ultralytics vì hue ít "
                 "ảnh hưởng tới chẩn đoán."
        )
        p.add_argument(
            "--hsv-s", type=float, default=0.3,
            help="Augmentation saturation jitter (fraction). Giảm từ default 0.7 xuống 0.3 "
                 "vì độ bão hòa màu lưỡi (đỏ/tím/nhạt) là tín hiệu chẩn đoán, không nên "
                 "jitter quá mạnh."
        )
        p.add_argument(
            "--hsv-v", type=float, default=0.2,
            help="Augmentation value/brightness jitter (fraction). Giảm từ default 0.4 "
                 "xuống 0.2 cùng lý do với hsv-s."
        )
        # ==== HẾT PHẦN THÊM MỚI ====

        p.add_argument("--save-period", type=int, default=20)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--name", type=str, default=None,
                       help="Experiment name (auto-generated as '<model>-imgsz<imgsz>-e<epochs>' if not set)")
        p.add_argument(
            "--exist-ok", action="store_true",
            help="Reuse an existing run folder instead of creating a fresh one. "
                 "Default is OFF, so every run gets its own folder (name, name2, name3, ...) "
                 "and never silently overwrites/merges a previous run's checkpoints."
        )
        p.add_argument(
            "--log-file", type=str, default=None,
            help="Path to save console log (default: <project>/<name>/train_<timestamp>.log)"
        )

    # Test args
    p_test.add_argument(
        "--data", type=str,
        default=str(DATA_YAML),
        help="Dataset YAML path"
    )
    p_test.add_argument("--imgsz", type=int, default=640)
    p_test.add_argument("--batch", type=int, default=16)
    p_test.add_argument("--device", type=str, default="0")
    p_test.add_argument("--project", type=str, default="runs/test")
    p_test.add_argument("--name", type=str, default=None)
    p_test.add_argument("--exist-ok", action="store_true",
                         help="Reuse an existing run folder instead of creating a fresh one.")
    p_test.add_argument("--workers", type=int, default=8)
    p_test.add_argument("--quiet", action="store_true")
    p_test.add_argument(
        "--log-file", type=str, default=None,
        help="Path to save console log"
    )

    args = parser.parse_args()

    # Auto-generate a per-model experiment name for 'single' (train_all already
    # does this internally) so checkpoints never land in a shared/default folder
    # just because --name was forgotten.
    if args.command == "single" and not args.name:
        args.name = f"{args.model}-imgsz{args.imgsz}-e{args.epochs}"

    # Set up logging
    if args.command in ("single", "all", "test") and hasattr(args, "log_file") and args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee = Tee(log_path)
        sys.stdout = tee
        print(f"Logging to {log_path.resolve()}")
    elif args.command in ("single", "all") and not args.log_file:
        # Auto-log inside the experiment's own folder, not the shared project root,
        # so logs from different models/runs don't pile up in one place.
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir_name = args.name if args.command == "single" else "all_runs"
        log_dir = Path(args.project) / exp_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"train_{ts}.log"
        tee = Tee(log_path)
        sys.stdout = tee
        print(f"Logging to {log_path.resolve()}")

    if args.command == "single":
        print(f"Checkpoints will be saved under: {Path(args.project) / args.name}")
        train_single(args)
    elif args.command == "all":
        train_all(args)
    elif args.command == "test":
        test_model(args)
    else:
        parser.print_help()

    # Restore stdout (Tee.close() already does this internally)
    if isinstance(sys.stdout, Tee):
        sys.stdout.close()


if __name__ == "__main__":
    main()
