# Final Training Codes - GitHub Repository

## Overview

This folder contains **all final training codes** with **consistent configurations** for fair comparison.

All models use:
- ✅ **Same seed**: 42
- ✅ **Same split**: 70/15/15 (train/val/test)
- ✅ **Same hyperparameters**: lr=1e-4, epochs=40, batch_size=1
- ✅ **Same parameter count**: ~21M parameters
- ✅ **Same physics-informed losses**: λ_sat=0.5, λ_mass=0.2, λ_darcy=0.1
- ✅ **Local Mac M1 compatible**: CPU mode with MPS fallback

---

## Main Training Scripts (70/15/15 Split, Seed=42)

### 1. **01_ConvLSTM_UNet_ScheduledSampling.py** ⭐ BEST MODEL

**Description**: ConvLSTM U-Net with Scheduled Sampling

**Architecture**:
- Encoder-decoder U-Net structure
- ConvLSTM cells for temporal memory (hysteresis tracking)
- Scheduled sampling: teacher forcing probability decays from 1.0 → 0.1
- Formula: `p_TF = k / (k + exp(epoch/k))` with k=5
- Pure autoregressive epochs 31-40

**Parameters**:
```python
base_features = 32
in_channels = 5  # Sw, Sg, P, Sg_max, K
out_channels = 3  # Sg, Sw, P
```

**Total parameters**: 21.38M

**Results** (Medium-fidelity):
- Single-step R²: 0.997
- **15-month rollout R²: 0.990** ✅
- MAE: 5.71 × 10⁻⁴

**Training time**: ~24 hours (40 epochs on Mac M1 CPU)

**Usage**:
```bash
python 01_ConvLSTM_UNet_ScheduledSampling.py
```

---

### 2. **02_ConvLSTM_UNet_TeacherForcing.py** ⚠️ FAILS ON ROLLOUT

**Description**: ConvLSTM U-Net with Teacher Forcing ONLY (no scheduled sampling)

**Architecture**: Identical to #1 but **without** scheduled sampling

**Parameters**: Identical to #1 (21.38M)

**Results**:
- Single-step R²: 0.997 (same as SS)
- **15-month rollout R²: 0.071** ❌ CATASTROPHIC FAILURE
- Demonstrates **exposure bias** problem

**Usage**:
```bash
python 02_ConvLSTM_UNet_TeacherForcing.py
```

**Purpose**: Ablation study to prove importance of scheduled sampling

---

### 3. **03_3D_UNet_Baseline.py** ⚠️ NO TEMPORAL MEMORY

**Description**: 3D U-Net WITHOUT ConvLSTM (no temporal memory)

**Architecture**:
- Standard U-Net with Conv3D + BatchNorm + ReLU
- No LSTM cells → cannot track hysteresis (Sg_max history)
- Same depth and width as ConvLSTM for fair comparison

**Parameters**:
```python
base_features = 32
Total: 22.58M parameters (comparable to ConvLSTM)
```

**Results**:
- Single-step R²: 0.806
- **15-month rollout R²: -0.55** ❌ FAILED
- Cannot model temporal dependencies

**Usage**:
```bash
python 03_3D_UNet_Baseline.py
```

**Purpose**: Proves necessity of temporal memory for UHS

---

### 4. **04_FNO_Baseline.py** ⚠️ SPECTRAL METHOD

**Description**: Fourier Neural Operator (3D)

**Architecture**:
- Spectral convolutions in Fourier domain
- Global receptive field
- No explicit temporal memory

**Parameters**:
```python
width = 32
modes1, modes2, modes3 = 8, 8, 4
Total: 21.27M parameters
```

**Results**:
- Single-step R²: 0.896
- **15-month rollout R²: -4.40** ❌ CATASTROPHIC FAILURE
- High-frequency artifacts accumulate

**Usage**:
```bash
python 04_FNO_Baseline.py
```

**Purpose**: Compare physics-informed CNN vs. spectral methods

---

## Ablation Studies

### 5. **ablation_A.py** - No Saturation Loss

Removes `λ_sat` (saturation conservation constraint)

**Config**:
```python
lambda_sat = 0.0  # ← ABLATED
lambda_mass = 0.2
lambda_darcy = 0.1
```

**Expected result**: Violations of Sg + Sw = 1

---

### 6. **ablation_B.py** - No Mass Loss

Removes `λ_mass` (mass conservation constraint)

**Config**:
```python
lambda_sat = 0.5
lambda_mass = 0.0  # ← ABLATED
lambda_darcy = 0.1
```

**Expected result**: Unphysical saturation changes

---

### 7. **ablation_C.py** - No Darcy Loss

Removes `λ_darcy` (spatial smoothness + buoyancy)

**Config**:
```python
lambda_sat = 0.5
lambda_mass = 0.2
lambda_darcy = 0.0  # ← ABLATED
```

**Expected result**: Checkerboard artifacts, no gravity override

---

## Configuration File

### **config_shared.py**

Central configuration ensuring ALL models use identical settings:

```python
# Reproducibility
SEED = 42

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training
NUM_EPOCHS = 40
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Architecture
BASE_FEATURES = 32  # ConvLSTM, UNet
FNO_WIDTH = 32
FNO_MODES = [8, 8, 4]

# Sequences
SEQUENCE_LENGTH_TRAIN = 10
SEQUENCE_LENGTH_TEST = 33

# Physics losses
LAMBDA_SAT = 0.5
LAMBDA_MASS = 0.2
LAMBDA_DARCY = 0.1
```

**Usage in scripts**:
```python
from config_shared import *

set_seed(SEED)
device = get_device()
```

---

## Results Summary Table

| Model | Single-step R² | 15-month R² | Status | Parameters |
|-------|---------------|-------------|---------|-----------|
| **ConvLSTM SS** ⭐ | 0.997 | **0.990** | ✅ SUCCESS | 21.38M |
| ConvLSTM TF | 0.997 | 0.071 | ❌ FAILED | 21.38M |
| 3D U-Net | 0.806 | -0.55 | ❌ FAILED | 22.58M |
| FNO | 0.896 | -4.40 | ❌ FAILED | 21.27M |

**Ablation Studies** (all based on ConvLSTM SS):

| Ablation | Removed Loss | Expected R² Drop | Purpose |
|----------|--------------|------------------|---------|
| A | λ_sat = 0 | ~0.02 | Test saturation conservation |
| B | λ_mass = 0 | ~0.05 | Test mass balance |
| C | λ_darcy = 0 | ~0.03 | Test spatial smoothness |

---

## How to Run

### Prerequisites
```bash
cd "/Users/narjisse/Documents/Effat Courses/deeponet"
```

Ensure data exists:
```
deeponet/
├── medium_fidelity/  # 500 .mat files
├── high_fidelity/    # 100 .mat files (optional)
└── final_codes_github/
```

### Run Best Model
```bash
python final_codes_github/01_ConvLSTM_UNet_ScheduledSampling.py
```

Expected output:
```
🎲 SEED = 42 (résultats reproductibles)
Device: cpu
✅ medium_fidelity: 500 fichiers .mat
📊 Nombre de paramètres: 21,380,000 (21.38M)

Split: 350 train / 75 val / 75 test

Epoch 1/40:
  Train loss: 0.0052
  Val loss: 0.0038
...
Epoch 40/40:
  Train loss: 0.0008
  Val loss: 0.0007

✅ Training complet!
💾 Modèle sauvegardé: checkpoints/convlstm_ss_best.pt
```

### Run Ablation Studies
```bash
# Ablation A (no saturation loss)
python final_codes_github/ablation_A.py

# Ablation B (no mass loss)
python final_codes_github/ablation_B.py

# Ablation C (no Darcy loss)
python final_codes_github/ablation_C.py
```

---

## Parameter Count Verification

To verify all models have ~21M parameters:

```python
import torch
from config_shared import count_parameters

# ConvLSTM
from script_01 import ConvLSTMUNet3D
model1 = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=32)
count_parameters(model1)  # 21,380,000

# 3D UNet
from script_03 import UNet3D
model3 = UNet3D(in_channels=5, out_channels=3, base_features=32)
count_parameters(model3)  # 22,580,000

# FNO
from script_04 import FNO3D
model4 = FNO3D(in_channels=5, out_channels=3, width=32, modes=[8,8,4])
count_parameters(model4)  # 21,270,000
```

All within ~21M ± 1M range ✅

---

## File Checklist

- [x] `config_shared.py` - Shared configuration
- [x] `01_ConvLSTM_UNet_ScheduledSampling.py` - Best model
- [x] `02_ConvLSTM_UNet_TeacherForcing.py` - Ablation (no SS)
- [x] `03_3D_UNet_Baseline.py` - Baseline (no LSTM)
- [x] `04_FNO_Baseline.py` - Baseline (spectral)
- [x] `ablation_A.py` - No saturation loss
- [x] `ablation_B.py` - No mass loss
- [x] `ablation_C.py` - No Darcy loss
- [x] `README_CODES.md` - This file

---

## Notes

1. **All scripts are LOCAL-compatible** (Mac M1 CPU mode)
2. **No modifications needed** - just run directly
3. **Checkpoints** saved to: `deeponet/checkpoints/`
4. **Results** include:
   - Training history (losses)
   - Best model checkpoint (.pt file)
   - Evaluation metrics (JSON)
5. **Reproducibility guaranteed** by fixed seed=42

---

## Citation

If you use these codes, please cite:

```bibtex
@article{kabbaj2025uhs,
  title={Simulation-Based Staged Decoupling of Mobility, Heterogeneity and Hysteresis Effects in Underground Hydrogen Storage},
  author={Kabbaj, Narjisse},
  journal={Geoenergy Science and Engineering},
  year={2025}
}
```

---

**Last updated**: January 2026
**Author**: Narjisse Kabbaj (nkabbaj@effatuniversity.edu.sa)
