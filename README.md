# CCS: A Continuous Spatial-Semantic Concordance Score for Robust Evaluation of Object Detection Models

[![Paper: SIVP (Springer)](https://img.shields.io/badge/Journal-Signal%2C%20Image%20and%20Video%20Processing-orange.svg)](https://www.springer.com/journal/11760)

This repository contains the official implementation, evaluation suite, and full experiment reproduction code for the paper:  
**"CCS: A Continuous Spatial-Semantic Concordance Score for Robust Evaluation of Object Detection Models"** (*Signal, Image and Video Processing*, Springer).

---

## 📌 Citation

If you find this work or metric useful in your research, please cite:

```bibtex
@article{mai2026ccs,
  title={CCS: A Continuous Spatial-Semantic Concordance Score for Robust Evaluation of Object Detection Models},
  author={Mai, Quoc Thai},
  journal={Signal, Image and Video Processing},
  publisher={Springer},
  year={2026}
}
```

---

## 📖 Overview & Theoretical Framework

Traditional object detection evaluation protocols (e.g., COCO mAP, F1-score) evaluate detection quality via disjoint binary gates:
1. **Localization Step-Discontinuity**: A predicted box is declared a match only if $\text{IoU} \ge \tau$ (e.g., 0.5), collapsing smooth spatial proximity into an abrupt binary decision.
2. **Semantic Equivalence Assumption**: Every incorrect label is treated as equally wrong ($0$ credit), discarding clinical and taxonomic proximity between adjacent symptoms (e.g., uniformly red tongue vs. red-spotted tongue).

**CCS (Continuous Concordance Score)** resolves both issues within a unified, continuous Hilbert-space framework:

$$\text{CCS} = \alpha \cdot C_{\text{sp}} + \beta \cdot C_{\text{sem}}$$

* **$C_{\text{sp}}$ (Spatial Concordance)**: Analytical closed-form Gaussian cosine similarity $s_k$ with exact scale-penalty and Gaussian distance decay (calibration-free, no hyperparameters).
* **$C_{\text{sem}}$ (Semantic Concordance)**: Taxonomy-driven Wu–Palmer similarity $\text{Sim}_{\text{WP}}$ grounded in canonical TCM clinical ontology ([WHO International Standard Terminologies, 2007](https://iris.who.int/handle/10665/207435)), awarding graded partial credit for near-miss label ambiguities.
* **$\text{CCS}_{\text{inst}}$ (Instance-Level Hungarian Formulation)**: Global optimal 1-to-1 bipartite assignment via Kuhn–Munkres optimization:
  $$\text{CCS}_{\text{inst}} = \frac{\sum_{i=1}^{|\pi^*|} S_{i, \pi^*(i)}}{\max(M, N)}, \quad S_{ij} = \alpha s_k(B_i, B_j) + \beta \text{Sim}_{\text{WP}}(c_i, c_j)$$

---

## 🚀 Environment Setup & Data Preparation

### 1. Installation

Clone this repository and install the dependencies:
```bash
git clone https://github.com/tmai110605/TongueTriage.git
cd TongueTriage
pip install -r requirements.txt
```

### 2. Dataset
The benchmark dataset uses the 8-class tongue-symptom subset of the TCM-Tongue dataset ($N=551$ test images), located in `shezhen datasets/shezhenv3-8class/`.

The 8 clinical symptom classes are:
- `0`: `botaishe` (Peeled / thin coating)
- `1`: `hongshe` (Red tongue body)
- `2`: `pangdashe` (Swollen tongue body)
- `3`: `hongdianshe` (Red-spotted petechiae)
- `4`: `liewenshe` (Fissured / cracked tongue)
- `5`: `chihenshe` (Teeth-marked tongue)
- `6`: `baitaishe` (White coating)
- `7`: `huangtaishe` (Yellow coating)

---

## 🔬 Reproducing Benchmark Experiments

All evaluation experiments and benchmarks can be executed with standalone scripts:

### 1. Unified Comprehensive Benchmark Suite
Runs the comprehensive evaluation suite covering:
* **Instance-Level Hungarian vs. Class-Level CCS**: Comparing granular bounding box assignment against diagnosis-level proxy across 6 YOLO models.
* **Confidence Threshold Sweep**: Evaluates model performance across confidence thresholds $\tau_{\text{conf}} \in [0.10, 0.90]$ and computes threshold-integrated $\text{mCCS}$ and $\text{AUC}_{\text{CCS}}$.
* **Taxonomy Sensitivity Analysis**: Evaluates rank stability across 4 taxonomic structures (Default 3-branch, Alt1 4-branch, Alt2 Syndromic, Alt3 2-level Flat).
* **Computational Latency & Throughput**: Benchmarks per-image latency and throughput (FPS) on CPU.
* **Statistical Significance & Effect Sizes**: Calculates pairwise Wilcoxon signed-rank $p$-values, Cohen's $d$, and Hedges' $g$.

```bash
python experiments_rebuttal.py
```
*Outputs saved to:* `runs/experiments/rebuttal_results.json` and `runs/experiments/fig_confidence_sweep.pdf`.

---

### 2. NMS Threshold Invariance & Annotation Noise Robustness
Evaluates metric invariance across NMS IoU thresholds $\tau_{\text{NMS}} \in [0.30, 0.80]$ and simulates human diagnostic variability:
* **Spatial Jitter**: Box centroid and scale perturbation from $0\%$ to $30\%$.
* **Semantic Label Corruption**: Random label noise from $0\%$ to $30\%$.

```bash
python exp_nms_and_noise.py
```
*Outputs saved to:* `runs/experiments/nms_and_noise_results.json`, `runs/experiments/fig_nms_sensitivity.pdf`, and `runs/experiments/fig_annotation_noise.pdf`.

---

### 3. Multi-Seed Bootstrap Resampling ($95\%$ Confidence Intervals)
Computes 1,000-iteration paired bootstrap resampling to quantify run variance and report $\text{Mean} \pm \text{Std}$ with $95\%$ Confidence Intervals:

```bash
python exp_bootstrap_ci.py
```
*Outputs saved to:* `runs/experiments/bootstrap_ci_results.json`.

---

### 4. Computational Latency & Throughput Benchmark (COCO mAP vs. CCS)
Benchmarks exact execution time, latency per image (ms), throughput (FPS), and time per 1,000 images on standard CPU:

```bash
python exp_map_time_benchmark.py
```

| Metric | Latency (ms / img) | Throughput (FPS) | Time (s / 1,000 imgs) |
|---|:---:|:---:|:---:|
| Standard IoU Matching ($\text{F1}@0.5$) | $0.007\text{ ms}$ | $142,850\text{ FPS}$ | $0.007\text{ s}$ |
| **Class-Level CCS (Closed-Form)** | **$0.011\text{ ms}$** | **$90,900\text{ FPS}$** | **$0.011\text{ s}$** |
| Full COCO $\text{mAP}@[.5:.95]$ (10 thresholds) | $0.124\text{ ms}$ | $8,060\text{ FPS}$ | $0.124\text{ s}$ |
| **Instance Hungarian CCS ($\text{CCS}_{\text{inst}}$)** | $0.271\text{ ms}$ | $3,690\text{ FPS}$ | $0.271\text{ s}$ |

---

### 5. Additional Detailed Analysis Scripts
* **CCS vs. mAP Ranking Discrepancy & Threshold Sensitivity**:
  ```bash
  python experiments_ccs_v2.py
  ```
* **Normalized Wasserstein Distance (NWD) Baseline**:
  ```bash
  python nwd_baseline.py
  ```
* **Weight Sensitivity ($\alpha, \beta$ Ablation & Optimization)**:
  ```bash
  python alpha_beta_optimization.py
  ```
* **Clinical Near-Miss & Far-Miss Classification**:
  ```bash
  python run_near_miss.py
  ```

---

## 🛠️ Model Training (YOLOv8, YOLOv10, YOLOv11)

To train or fine-tune detectors on the 8-class benchmark:

```bash
# Train a single model (e.g. YOLOv10m)
python train_8class.py single --model yolov10m.pt --epochs 100 --imgsz 1024 --batch 16 --device 0

# Train all 6 models sequentially
python train_8class.py all --models yolov8n,yolov8m,yolov10n,yolov10m,yolov11n,yolov11m --epochs 100 --imgsz 1024 --batch 16 --device 0
```