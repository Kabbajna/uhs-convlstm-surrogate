# Final Codes Index - Complete Overview

📁 **Location**: `/Users/narjisse/Documents/Effat Courses/deeponet/final_codes_github/`

---

## ✅ All Files Ready for GitHub

### Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `config_shared.py` | Shared configuration (SEED, splits, hyperparameters) | ✅ Ready |
| `verify_consistency.py` | Verification script to check all configs match | ✅ Ready |
| `README_CODES.md` | Main documentation for this folder | ✅ Ready |
| `INDEX.md` | This file | ✅ Ready |

### Main Training Scripts (4 models for comparison)

| File | Model | R² (15-month) | Parameters | Status |
|------|-------|---------------|------------|--------|
| `01_ConvLSTM_UNet_ScheduledSampling.py` | ConvLSTM SS ⭐ | **0.990** | 21.38M | ✅ Ready |
| `02_ConvLSTM_UNet_TeacherForcing.py` | ConvLSTM TF | 0.071 ❌ | 21.38M | ✅ Ready |
| `03_3D_UNet_Baseline.py` | 3D U-Net | -0.55 ❌ | 22.58M | ✅ Ready |
| `04_FNO_Baseline.py` | FNO 3D | -4.40 ❌ | 21.27M | ✅ Ready |

### Ablation Studies (3 studies)

| File | What's Ablated | Expected Impact | Status |
|------|----------------|-----------------|--------|
| `ablation_A.py` | λ_sat = 0 (no saturation conservation) | Sg + Sw ≠ 1 violations | ✅ Ready |
| `ablation_B.py` | λ_mass = 0 (no mass conservation) | Unphysical saturation changes | ✅ Ready |
| `ablation_C.py` | λ_darcy = 0 (no Darcy constraint) | Spatial artifacts | ✅ Ready |

---

## Consistency Verification Results

```bash
cd final_codes_github
python3 verify_consistency.py
```

**Output**:
```
✅ SEED = 42 pour tous les modèles
✅ Split 70/15/15 pour tous les modèles
✅ Learning rate = 1e-4 pour tous les modèles
✅ Epochs = 40 pour tous les modèles
✅ Base features = 32 pour tous les modèles
✅ Tous les fichiers présents
```

---

## Shared Configuration Summary

All 7 Python files use these **IDENTICAL** configurations:

```python
SEED = 42
TRAIN_SPLIT = 0.70  # 70%
VAL_SPLIT = 0.15    # 15%
TEST_SPLIT = 0.15   # 15%

NUM_EPOCHS = 40
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

BASE_FEATURES = 32  # For ConvLSTM and UNet
FNO_WIDTH = 32
FNO_MODES = [8, 8, 4]

SEQUENCE_LENGTH_TRAIN = 10
SEQUENCE_LENGTH_TEST = 33

LAMBDA_SAT = 0.5    # Saturation conservation
LAMBDA_MASS = 0.2   # Mass conservation
LAMBDA_DARCY = 0.1  # Darcy flow constraint
```

---

## Quick Start Guide

### 1. Run Best Model (ConvLSTM with Scheduled Sampling)

```bash
cd "/Users/narjisse/Documents/Effat Courses/deeponet/final_codes_github"
python3 01_ConvLSTM_UNet_ScheduledSampling.py
```

**Expected runtime**: ~24 hours on Mac M1 CPU

**Output**:
- Training/validation losses per epoch
- Final model: `../checkpoints/convlstm_ss_best.pt`
- Results: R² = 0.990 on 15-month rollout

### 2. Run All Baselines

```bash
# Teacher Forcing only (fails on rollout)
python3 02_ConvLSTM_UNet_TeacherForcing.py

# 3D U-Net without temporal memory
python3 03_3D_UNet_Baseline.py

# FNO spectral method
python3 04_FNO_Baseline.py
```

### 3. Run Ablation Studies

```bash
# Remove saturation loss
python3 ablation_A.py

# Remove mass loss
python3 ablation_B.py

# Remove Darcy loss
python3 ablation_C.py
```

---

## File Sizes

```
Total size: ~260 KB (Python scripts only)

01_ConvLSTM_UNet_ScheduledSampling.py    33 KB  (main model)
02_ConvLSTM_UNet_TeacherForcing.py       28 KB
03_3D_UNet_Baseline.py                   26 KB
04_FNO_Baseline.py                       31 KB
ablation_A.py                            33 KB
ablation_B.py                            33 KB
ablation_C.py                            33 KB
config_shared.py                          5 KB
verify_consistency.py                     5 KB
README_CODES.md                           8 KB
```

---

## What Makes These Codes "GitHub-Ready"?

✅ **Consistent configurations** - All use same SEED, split, hyperparameters
✅ **Same parameter count** - All ~21M parameters for fair comparison
✅ **Well documented** - Clear README with results tables
✅ **Verified** - Automated verification script confirms consistency
✅ **Self-contained** - Each script runs independently
✅ **Local-compatible** - Mac M1 CPU mode, no GPU needed
✅ **Reproducible** - Fixed seed=42, deterministic training

---

## Expected Results Table

| Model | Architecture | Single-step R² | 15-month R² | Inference (33 steps) |
|-------|-------------|----------------|-------------|---------------------|
| ConvLSTM SS ⭐ | U-Net + LSTM + Scheduled Sampling | 0.997 | **0.990** ✅ | 1.5 sec |
| ConvLSTM TF | U-Net + LSTM + Teacher Forcing | 0.997 | 0.071 ❌ | 1.5 sec |
| 3D U-Net | U-Net without LSTM | 0.806 | -0.55 ❌ | 1.2 sec |
| FNO | Spectral method | 0.896 | -4.40 ❌ | 2.1 sec |

**Key findings**:
1. ✅ **Scheduled sampling is critical** (0.990 vs 0.071)
2. ✅ **Temporal memory is essential** (ConvLSTM vs plain UNet)
3. ✅ **Spatial CNN > spectral methods** for this problem

---

## Next Steps for GitHub

1. ✅ All codes created with consistent configs
2. ✅ Verification script confirms consistency
3. ✅ Documentation (README) complete
4. ⏭️ Copy this folder to your GitHub repository:

```bash
# Initialize git (if not done)
cd "/Users/narjisse/Documents/Effat Courses/deeponet"
git add final_codes_github/
git commit -m "Add final training codes with consistent configurations"
git push origin main
```

---

## Contact

**Narjisse Kabbaj**
Energy Research Lab, College of Engineering
Effat University, Jeddah, Saudi Arabia
Email: nkabbaj@effatuniversity.edu.sa

---

**Last updated**: January 1, 2026
