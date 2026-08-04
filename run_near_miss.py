import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments_ccs_v2 import (
    load_test_paths,
    _eval_one_model,
    DATA_YAML,
    OUTPUT_DIR,
    CHECKPOINTS,
)

from ccs import build_semantic_matrix
from near_miss_analysis import (
    analyze_near_miss,
    save_near_miss_report,
)


def main():

    sem_matrix = build_semantic_matrix()

    # Use CHECKPOINTS directly
    checkpoints = CHECKPOINTS

    print("\n" + "=" * 70)
    print("CHECKPOINTS TO RUN")
    print("=" * 70)

    for model_name, checkpoint_path in checkpoints.items():
        print(f"{model_name}: {checkpoint_path}")
        print(f"Exists: {Path(checkpoint_path).exists()}")

    img_paths, label_dir = load_test_paths(DATA_YAML)

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"Number of images: {len(img_paths)}")
    print(f"Label directory: {label_dir}")

    all_results = {}

    for model_name in checkpoints:

        print("\n" + "=" * 70)
        print(f"RUNNING MODEL: {model_name}")
        print("=" * 70)

        per_image = _eval_one_model(
            checkpoints,
            img_paths,
            label_dir,
            sem_matrix,
            model_name,
        )

        all_results[model_name] = {
            "per_image": per_image
        }

    summary_rows, pooled_pct = analyze_near_miss(
        all_results
    )

    save_near_miss_report(
        summary_rows,
        pooled_pct,
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()