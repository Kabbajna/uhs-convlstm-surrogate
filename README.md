# Underground Hydrogen Storage - ConvLSTM-UNet Surrogate Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the implementation of the ConvLSTM-UNet surrogate model for Underground Hydrogen Storage (UHS) simulation, as presented in:

**"Simulation-Based Staged Decoupling of Mobility, Heterogeneity and Hysteresis Effects in Underground Hydrogen Storage"**
*Narjisse Kabbaj*
Energy Research Lab, College of Engineering, Effat University, Jeddah, Saudi Arabia
Submitted to *Geoenergy Science and Engineering*, December 2025

## Overview

This work presents a physics-informed deep learning framework that accelerates Underground Hydrogen Storage predictions by **119-200×** compared to traditional numerical simulation (MRST), while maintaining **R² = 0.99** accuracy over 15-month autoregressive rollouts.

### Key Features

- **3D Spatio-temporal Prediction**: Full 3D reservoir simulation with ConvLSTM-UNet architecture
- **Hysteresis Modeling**: Explicit tracking of drainage-imbibition cycles via S_g,max
- **Scheduled Sampling**: Mitigates exposure bias for long-horizon autoregressive rollout
- **Resolution Transfer**: Zero-shot generalization from 20³ to 40×40×20 grids (R² = 0.968)
- **Physics-Informed Loss**: Enforces saturation conservation, mass balance, and Darcy flow constraints
- **Computational Speedup**: 119× (medium-fidelity) to 200× (high-fidelity) acceleration

## Architecture

The ConvLSTM-UNet combines:
- **U-Net encoder-decoder** for multi-scale spatial feature extraction
- **ConvLSTM cells** for temporal memory and hysteresis tracking
- **Skip connections** preserving fine-grained spatial features
- **21.38M parameters** trained on 500 medium-fidelity + 100 high-fidelity MRST simulations

### Model Input/Output

**Input** (5 channels):
- Water saturation (S_w)
- Gas saturation (S_g)
- Pressure (P, normalized)
- Historical maximum gas saturation (S_g,max)
- Permeability (K, normalized)

**Output** (3 channels):
- Predicted gas saturation (S_g)
- Predicted water saturation (S_w)
- Predicted pressure (P)

## Results Summary

| Metric | Medium-Fidelity (20³) | High-Fidelity (40×40×20) |
|--------|----------------------|--------------------------|
| **Rollout R²** | 0.990 ± 0.004 | 0.987 ± 0.002 (zero-shot) |
| **MAE** | 5.71 × 10⁻⁴ | 6.2 × 10⁻⁴ |
| **Inference Time** | 1.5 seconds | 4.5 seconds |
| **MRST Time** | 3.0 minutes | 15.0 minutes |
| **Speedup** | **119×** | **200×** |

### Ablation Study

| Model | Single-step R² | 15-month Rollout R² |
|-------|---------------|---------------------|
| ConvLSTM-UNet (Scheduled Sampling) | 0.997 | **0.990** |
| ConvLSTM-UNet (Teacher Forcing) | 0.997 | 0.071 (failed) |
| 3D U-Net | 0.806 | −0.55 (failed) |
| FNO | 0.896 | −4.40 (failed) |

## Repository Structure

```
deeponet/
├── README.md                           # This file
├── LICENSE                             # License information
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
│
├── data/                              # Dataset (not included - see Data Availability)
│   ├── low_fidelity/                  # 2000 simulations (reference only)
│   ├── medium_fidelity/               # 500 simulations (20×20×20)
│   └── high_fidelity/                 # 100 simulations (40×40×20)
│
├── src/                               # Source code
│   ├── models/                        # Model architectures
│   │   ├── convlstm_unet.py          # Main ConvLSTM-UNet model
│   │   ├── baseline_unet.py          # 3D U-Net baseline
│   │   └── fno_3d.py                 # FNO baseline
│   ├── data/                          # Data loading and preprocessing
│   │   ├── dataset.py                # PyTorch dataset classes
│   │   └── preprocessing.py          # Normalization utilities
│   ├── training/                      # Training utilities
│   │   ├── trainer.py                # Training loop
│   │   ├── losses.py                 # Physics-informed loss functions
│   │   └── scheduled_sampling.py     # Scheduled sampling implementation
│   └── evaluation/                    # Evaluation and metrics
│       ├── metrics.py                # R², MAE, recovery factor
│       └── visualization.py          # Plotting utilities
│
├── scripts/                           # Executable scripts
│   ├── train_medium_fidelity.py      # Train on 20³ grid
│   ├── train_high_fidelity.py        # Fine-tune on 40×40×20 grid
│   ├── evaluate_model.py             # Evaluation script
│   └── generate_mrst_data.m          # MATLAB/MRST data generation
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset analysis
│   ├── 02_model_training.ipynb       # Training demonstration
│   └── 03_results_visualization.ipynb # Results and figures
│
├── checkpoints/                       # Model checkpoints
│   ├── convlstm_unet_medium.pt       # Best medium-fidelity model
│   └── convlstm_unet_high.pt         # Best high-fidelity model
│
├── results/                           # Experimental results
│   ├── figures/                       # Publication figures
│   ├── metrics/                       # Evaluation metrics (JSON)
│   └── ablation/                      # Ablation study results
│
└── docs/                              # Documentation
    ├── METHODOLOGY.md                 # Detailed methodology
    ├── MRST_SETUP.md                 # MRST simulation setup
    └── REPRODUCING.md                # Reproduction instructions
```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- MATLAB with MRST (for data generation only)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/uhs-convlstm-surrogate.git
cd uhs-convlstm-surrogate
```

2. Create a conda environment:
```bash
conda env create -f environment.yml
conda activate uhs-surrogate
```

Or install via pip:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

The complete dataset is available at [Zenodo/Figshare link]. Download and extract to the `data/` directory.

Alternatively, generate your own data using MRST:
```matlab
cd scripts
matlab -r "run generate_mrst_data.m"
```

### 2. Training

Train the ConvLSTM-UNet on medium-fidelity data:
```bash
python scripts/train_medium_fidelity.py \
    --data_path data/medium_fidelity \
    --batch_size 1 \
    --epochs 40 \
    --lr 1e-4 \
    --scheduled_sampling_k 5 \
    --output_dir checkpoints/
```

Fine-tune on high-fidelity data:
```bash
python scripts/train_high_fidelity.py \
    --data_path data/high_fidelity \
    --pretrained checkpoints/convlstm_unet_medium.pt \
    --epochs 20 \
    --lr 1e-5 \
    --output_dir checkpoints/
```

### 3. Evaluation

Evaluate trained model on test set:
```bash
python scripts/evaluate_model.py \
    --model_path checkpoints/convlstm_unet_medium.pt \
    --data_path data/medium_fidelity/test \
    --output_dir results/
```

### 4. Inference

Use the trained model for predictions:
```python
import torch
from src.models.convlstm_unet import ConvLSTMUNet
from src.data.dataset import UHSDataset

# Load model
model = ConvLSTMUNet(in_channels=5, out_channels=3)
model.load_state_dict(torch.load('checkpoints/convlstm_unet_medium.pt'))
model.eval()

# Load initial condition
initial_state = ...  # Shape: (1, 5, 20, 20, 20)

# Autoregressive rollout for 33 steps (15 months)
predictions = []
hidden_state = None

for step in range(33):
    with torch.no_grad():
        output, hidden_state = model(initial_state, hidden_state)
    predictions.append(output)

    # Update input for next step
    initial_state = prepare_next_input(output)

# predictions now contains full 15-month trajectory
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `batch_size` | 1 | Due to 3D memory requirements |
| `epochs` | 40 (medium), 20 (high) | Training duration |
| `scheduled_sampling_k` | 5 | Decay parameter for teacher forcing |
| `lambda_sat` | 0.5 | Saturation loss weight |
| `lambda_mass` | 0.2 | Mass conservation loss weight |
| `lambda_darcy` | 0.1 | Darcy flow loss weight |
| `sequence_length_train` | 10 | Training sequence length |
| `sequence_length_test` | 33 | Inference rollout length (15 months) |

## Physical Parameters

The simulations use Brooks-Corey hysteresis parameters fitted to experimental H₂-brine data:

| Parameter | Drainage | Imbibition |
|-----------|----------|------------|
| Connate water saturation (S_wc) | 0.45 | 0.45 |
| Residual gas saturation (S_gr) | 0.02 | 0.24 |
| Water Corey exponent (n_w) | 4.01 | 4.00 |
| Gas Corey exponent (n_g) | 4.66 | 2.80 |
| Max. water rel. perm. (k_rw^max) | 0.95 | 0.90 |
| Max. gas rel. perm. (k_rg^max) | 0.20 | 0.17 |

**Reservoir conditions**: P ≈ 14.8 MPa, T = 55°C, depth = 1500 m

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{kabbaj2025uhs,
  title={Simulation-Based Staged Decoupling of Mobility, Heterogeneity and Hysteresis Effects in Underground Hydrogen Storage},
  author={Kabbaj, Narjisse},
  journal={Geoenergy Science and Engineering},
  year={2025},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Narjisse Kabbaj**
Energy Research Lab, College of Engineering
Effat University, Jeddah 22332, Saudi Arabia
Email: nkabbaj@effatuniversity.edu.sa

## Acknowledgments

- MATLAB Reservoir Simulation Toolbox (MRST) for data generation
- Experimental H₂-brine hysteresis data from Boon et al. (2022)
- Computational resources provided by Effat University

## Data Availability

Due to file size constraints, the complete dataset (500 + 100 simulations, ~50 GB) is hosted separately:

- **Medium-fidelity dataset**: [Link to be added]
- **High-fidelity dataset**: [Link to be added]
- **MRST generation scripts**: Available in `scripts/generate_mrst_data.m`

For access to the full dataset, please contact the corresponding author.

## Reproducibility

This repository includes:
- ✅ Complete source code
- ✅ Trained model checkpoints
- ✅ Hyperparameter configurations
- ✅ Evaluation scripts
- ✅ Random seed (42) for reproducibility
- ✅ Requirements files for exact environment

All experiments were run on Apple M1 CPU with 16 GB unified memory. Training takes approximately 24 hours for 40 epochs on medium-fidelity data.

---

**Last updated**: January 2026
