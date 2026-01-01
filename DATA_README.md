# Dataset Description

This document describes the multi-fidelity dataset used for training and evaluating the ConvLSTM-UNet surrogate model for Underground Hydrogen Storage.

## Dataset Overview

| Fidelity Level | Grid Size | # Simulations | # Cells | Sim Time | Total Size |
|----------------|-----------|---------------|---------|----------|------------|
| Low | 20×20×20 | 2000 | 8,000 | ~3 min | ~40 GB |
| Medium | 20×20×20 | 500 | 8,000 | ~3 min | ~10 GB |
| High | 40×40×20 | 100 | 32,000 | ~15 min | ~8 GB |

**Total dataset size**: ~58 GB (uncompressed)

## Data Generation

All data was generated using the MATLAB Reservoir Simulation Toolbox (MRST) with:
- **Simulator**: GenericBlackOilModel (two-phase water-gas flow)
- **Hysteresis model**: Brooks-Corey with Killough hysteresis
- **Parameters**: Fitted to experimental H₂-brine data from Boon et al. (2022)
- **Time discretization**: 34 timesteps at 13.2-day intervals (15 months total)
- **Operational sequence**: 2 complete injection-withdrawal cycles

### Parameter Space (Latin Hypercube Sampling)

| Parameter | Symbol | Range | Unit | Distribution |
|-----------|--------|-------|------|--------------|
| Permeability | K | [50, 500] | mD | Log-uniform |
| Porosity | φ | [0.15, 0.35] | - | Uniform |
| Rate multiplier | Q_mult | [0.5, 1.5] | - | Uniform |
| Cycle duration | τ | [150, 210] | days | Uniform |
| Hysteresis factor | η | [0.8, 1.2] | - | Uniform |

### Reservoir Configuration

- **Domain**: 1000 m × 1000 m × 100 m
- **Depth**: 1500 m (P ≈ 14.8 MPa, T = 55°C)
- **Well**: Single vertical well at domain center
- **Perforation**: Upper half of reservoir (50 m)
- **Boundaries**: No-flow lateral, constant-pressure bottom

## Data Format

Each simulation is stored as an HDF5 file with the following structure:

```
simulation_XXXXX.h5
├── /states/                     # Time-varying 3D fields
│   ├── Sw [34, Nz, Ny, Nx]     # Water saturation
│   ├── Sg [34, Nz, Ny, Nx]     # Gas saturation
│   ├── P  [34, Nz, Ny, Nx]     # Pressure (Pa)
│   └── Sg_max [34, Nz, Ny, Nx] # Historical max gas saturation
├── /properties/                 # Static reservoir properties
│   ├── K [Nz, Ny, Nx]          # Permeability (m²)
│   ├── phi [Nz, Ny, Nx]        # Porosity (-)
│   └── depth [Nz, Ny, Nx]      # Cell depth (m)
├── /operational/                # Well controls
│   ├── time [34]               # Timesteps (days)
│   ├── rate [34]               # Injection rate (m³/day)
│   └── phase [34]              # Operational phase (0=inj, 1=prod)
└── /metadata/                   # Simulation parameters
    ├── permeability_value
    ├── porosity_value
    ├── rate_multiplier
    ├── cycle_duration
    └── hysteresis_factor
```

### Reading Data Example

```python
import h5py
import numpy as np

# Load a simulation
with h5py.File('data/medium_fidelity/simulation_00001.h5', 'r') as f:
    # Load states
    Sw = f['states/Sw'][:]        # Shape: (34, 20, 20, 20)
    Sg = f['states/Sg'][:]
    P = f['states/P'][:]
    Sg_max = f['states/Sg_max'][:]

    # Load properties
    K = f['properties/K'][:]       # Shape: (20, 20, 20)
    phi = f['properties/phi'][:]

    # Load operational data
    time = f['operational/time'][:] # Shape: (34,)
    rate = f['operational/rate'][:]

    # Load metadata
    K_value = f['metadata/permeability_value'][()]
    phi_value = f['metadata/porosity_value'][()]
```

## Data Splits

### Medium-Fidelity (500 simulations)
- **Training**: 350 simulations (70%)
- **Validation**: 75 simulations (15%)
- **Test**: 75 simulations (15%)
- **Random seed**: 42

### High-Fidelity (100 simulations)
Used exclusively for zero-shot transfer evaluation:
- **Zero-shot test**: 20 simulations
- **Fine-tuning train**: 64 simulations (80% of 80)
- **Fine-tuning test**: 16 simulations (20% of 80)

## Preprocessing

### Normalization

All inputs are normalized before feeding to the model:

```python
# Saturation: already in [0, 1], no normalization needed
Sw_norm = Sw
Sg_norm = Sg
Sg_max_norm = Sg_max

# Pressure: normalize by reference pressure
P_ref = 14.8e6  # Pa
P_norm = P / P_ref

# Permeability: normalize by reference permeability
K_ref = 1e-13  # m² (100 mD)
K_norm = K / K_ref
```

### Data Augmentation

No spatial augmentation (flipping, rotation) was applied to preserve physical well location and gravity direction.

## Physical Validity

All simulations satisfy the following physical constraints:
- ✅ Saturation bounds: S_w, S_g ∈ [0, 1]
- ✅ Volume conservation: S_w + S_g = 1
- ✅ Hysteresis consistency: S_g ≤ S_g,max during imbibition
- ✅ Mass conservation: Total fluid volume preserved
- ✅ Pressure positivity: P > 0

## Downloading the Dataset

Due to the large size, the dataset is hosted on external platforms:

### Option 1: Zenodo (Recommended)
```bash
# Download medium-fidelity dataset
wget https://zenodo.org/record/[ID]/files/medium_fidelity.tar.gz
tar -xzf medium_fidelity.tar.gz -C data/

# Download high-fidelity dataset
wget https://zenodo.org/record/[ID]/files/high_fidelity.tar.gz
tar -xzf high_fidelity.tar.gz -C data/
```

### Option 2: Request from Author
For large-scale users or commercial applications, please contact:
**Dr. Narjisse Kabbaj** - nkabbaj@effatuniversity.edu.sa

## Generating Your Own Data

If you prefer to generate custom data:

1. Install MATLAB and MRST:
```matlab
% Download MRST from www.sintef.no/mrst
addpath('/path/to/mrst')
startup  % Initialize MRST
```

2. Run the generation script:
```bash
cd scripts
matlab -batch "run('generate_mrst_data.m')"
```

3. Customize parameters in `generate_mrst_data.m`:
   - Grid resolution
   - Parameter ranges
   - Number of simulations
   - Operational schedule

**Note**: Generating 500 medium-fidelity simulations takes approximately **25 hours** on a standard workstation.

## Data Quality Checks

Before using the data, verify integrity:

```python
from src.data.validation import validate_dataset

# Check dataset integrity
validate_dataset('data/medium_fidelity', verbose=True)
```

Expected output:
```
✅ All simulations have complete timesteps (34)
✅ No NaN or infinite values detected
✅ Physical constraints satisfied (Sg + Sw = 1)
✅ Hysteresis consistency verified (Sg ≤ Sg_max)
✅ Dataset ready for training
```

## License and Usage

This dataset is released under **CC BY 4.0** (Creative Commons Attribution 4.0 International):
- ✅ You may use, share, and adapt the data
- ✅ You must give appropriate credit and cite the original paper
- ✅ Commercial use is permitted

## Citation

If you use this dataset, please cite:

```bibtex
@article{kabbaj2025uhs,
  title={Simulation-Based Staged Decoupling of Mobility, Heterogeneity and Hysteresis Effects in Underground Hydrogen Storage},
  author={Kabbaj, Narjisse},
  journal={Geoenergy Science and Engineering},
  year={2025},
  note={Submitted}
}
```

## Contact

For questions about the dataset:
- **Technical issues**: Open an issue on GitHub
- **Data access**: nkabbaj@effatuniversity.edu.sa
- **Collaboration**: nkabbaj@effatuniversity.edu.sa
