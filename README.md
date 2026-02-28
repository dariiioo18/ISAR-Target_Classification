# ISAR Target Classification

Automatic Radar Target Recognition (ATR) using **ISAR images** generated from electromagnetic simulations (**NewFASANT**) and classified with machine-learning models and a CNN.

<p align="center">
  <img src="docs/isar_example.png" alt="Example ISAR image — normalised reflectivity surface" width="520">
  <br>
  <em>Example ISAR image: normalised reflectivity of a target obtained via 2-D FFT.</em>
</p>

---

## Overview

This project covers the **full pipeline** from raw electromagnetic simulation outputs to classification results:

1. **ISAR image generation** (MATLAB) — Parse NewFASANT RCS data, apply 2-D FFT to obtain ISAR images, and augment them with controllable Gaussian noise levels.
2. **Target classification** (Python) — Compare SVM, Decision Tree, Random Forest, K-Means (adapted), and a Convolutional Neural Network on the generated image dataset.

### Target classes

| Class | Description |
|-------|-------------|
| Caja | Rectangular box |
| Cilindro | Cylinder |
| Cono | Cone |

### Key results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM (RBF) | 0.794 | 0.796 |
| Decision Tree | 0.688 | 0.690 |
| Random Forest | 0.781 | 0.783 |
| K-Means | 0.401 | 0.397 |
| **CNN** | **0.758** | **0.761** |

---

## Project Structure

```
ISAR-Target-Classification/
├── matlab/
│   ├── isar_fft.m                  # 2-D FFT ISAR imaging
│   ├── parse_newfasant.m           # NewFASANT output parser
│   └── batch_isar_generator.m      # Batch image generator
├── python/
│   ├── config.py          # Hyperparameters and paths
│   ├── data_loader.py     # Image loading and preprocessing
│   ├── ml_models.py       # SVM, Decision Tree, Random Forest, K-Means
│   ├── cnn_model.py       # CNN definition and training
│   ├── evaluation.py      # Metrics and visualisation
│   └── main.py            # CLI entry point
├── docs/
│   └── isar_example.png   # Sample ISAR image for documentation
├── data/                  # Place your ISAR images here (not tracked)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- **MATLAB** R2020a or later (for image generation only)
- **Python** 3.9+
- A GPU is recommended for CNN training but not required

### Installation

```bash
git clone https://github.com/<your-username>/ISAR-Target-Classification.git
cd ISAR-Target-Classification
pip install -r requirements.txt
```

### 1. Generate ISAR images (MATLAB)

If you have NewFASANT simulation outputs, set the `ISAR_DATASETS_ROOT` environment variable to your dataset path (or let the script prompt you with a folder picker), then run:

```matlab
batch_isar_generator
```

Alternatively, set the environment variable before launching MATLAB:

```bash
# Linux / macOS
export ISAR_DATASETS_ROOT=/path/to/your/dataset

# Windows (cmd)
set ISAR_DATASETS_ROOT=C:\path\to\your\dataset
```

This will create PNG images organised by class and noise level.

### 2. Run the classification pipeline (Python)

Place your ISAR images (all classes mixed in one flat folder) in the `data/` directory, then:

```bash
cd python
python main.py --data ../data
```

#### CLI options

| Flag | Description |
|------|-------------|
| `--data PATH` | Path to the image folder (default: `../data`) |
| `--grid-search` | Run hyper-parameter grid search before training |
| `--skip-cnn` | Skip CNN training (ML models only) |

#### Examples

```bash
# Full pipeline with grid search
python main.py --data ../data --grid-search

# Quick ML-only evaluation (no CNN)
python main.py --data ../data --skip-cnn
```

---

## MATLAB Functions

### `isar_fft(G, freqs, angles)`

Generates an ISAR image via 2-D FFT from a complex scattering matrix.

**Parameters:**
- `Nfft` — FFT zero-padding size (default: 32)
- `PlotResult` — Display the surface plot (default: false)

### `parse_newfasant(filename, sigma)`

Parses a NewFASANT `RcsFieldRP.out` file and produces an ISAR image with optional Gaussian noise.

### `batch_isar_generator`

Batch script that iterates over the full simulation dataset and generates ISAR images at multiple noise levels (σ = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5). The dataset path is configured via the `ISAR_DATASETS_ROOT` environment variable or selected interactively through a folder picker dialog.

---

## Dataset

> **The dataset is not included in this repository for confidentiality reasons.**

The images used for classification are ISAR (Inverse Synthetic Aperture Radar) representations of the normalised reflectivity of each target. They are obtained by simulating the electromagnetic response of previously modelled 3-D geometries in **NewFASANT** and then applying a 2-D FFT post-processing step (see the MATLAB scripts above). Each image has the structure shown in the example at the top of this README.

To add robustness, each clean image is augmented with additive complex Gaussian noise at several levels (σ = 0.0, 0.1, 0.2, 0.3, 0.4, 0.5), resulting in ~8 000 images across the three target classes.

If you wish to generate your own dataset:

1. Model the target geometries and run RCS simulations in **NewFASANT**.
2. Set the `ISAR_DATASETS_ROOT` environment variable to point to your simulation output directory.
3. Execute `batch_isar_generator` in MATLAB — it will produce all PNG images automatically.

---

## Citation

If you use this work, please cite:

```
del Saz, D. (2025). ISAR Target Classification using Machine Learning and CNNs.
Bachelor's Thesis, Universidad de Alcalá.
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
