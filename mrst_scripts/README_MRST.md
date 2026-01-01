# MRST Data Generation Scripts

This folder contains MATLAB scripts for generating the multi-fidelity Underground Hydrogen Storage simulation data using MATLAB Reservoir Simulation Toolbox (MRST).

## Contents

- **generate_multifidelity_data.m** - Main script to generate all three fidelity levels

## Prerequisites

### Software Requirements
- MATLAB R2020a or later
- MATLAB Reservoir Simulation Toolbox (MRST)
  - Download from: https://www.sintef.no/projectweb/mrst/

### MRST Installation

1. Download MRST from the official website
2. Extract to a folder (e.g., `/path/to/mrst-2023b`)
3. Add to MATLAB path:
```matlab
addpath('/path/to/mrst-2023b')
startup  % Initialize MRST
```

## Data Generation Overview

The script generates **three fidelity levels**:

| Fidelity | Grid Size | # Simulations | Time per Sim | Total Time |
|----------|-----------|---------------|--------------|------------|
| **Low** | 20×20×20 | 2000 | ~3 min | ~100 hours |
| **Medium** | 20×20×20 | 500 | ~3 min | ~25 hours |
| **High** | 40×40×20 | 100 | ~15 min | ~25 hours |

**Total dataset**: ~150 hours of computation, ~50 GB

## How to Run

### Step 1: Configure Paths

Edit `generate_multifidelity_data.m` and set your output directory:

```matlab
% Change this to your desired output location
base_dir = '/path/to/output/multifidelity_deeponet_data';
```

### Step 2: Run the Script

```matlab
run generate_multifidelity_data.m
```

The script will:
1. Create output directories (high_fidelity, medium_fidelity, low_fidelity)
2. Generate Latin Hypercube samples for parameters
3. Run MRST simulations for each parameter set
4. Save results as `.mat` files

### Step 3: Monitor Progress

Check the log files:
```
logs/master_log_YYYYMMDD_HHMMSS.txt
logs/google_drive_sync_status.txt
```

## Parameter Ranges (Latin Hypercube Sampling)

| Parameter | Symbol | Range | Unit | Distribution |
|-----------|--------|-------|------|--------------|
| Permeability | K | [50, 500] | mD | Log-uniform |
| Porosity | φ | [0.15, 0.35] | - | Uniform |
| Rate multiplier | Q_mult | [0.5, 1.5] | - | Uniform |
| Cycle duration | τ | [150, 210] | days | Uniform |
| Hysteresis factor | η | [0.8, 1.2] | - | Uniform |

## Output Format

Each simulation is saved as:
```
fidelity_level/simulation_XXXXX.mat
```

Containing:
```matlab
sim_data:
  .timesteps (1×34 struct array)
    .Sw       [Nz, Ny, Nx]  Water saturation
    .Sg       [Nz, Ny, Nx]  Gas saturation
    .P        [Nz, Ny, Nx]  Pressure (Pa)
    .Sg_max   [Nz, Ny, Nx]  Historical max gas saturation
    .time     (scalar)      Time in days
    .Q        (scalar)      Injection rate
    .operation (string)     'injection' or 'withdrawal'
  .params
    .K        (scalar)      Permeability (m²)
    .phi      (scalar)      Porosity
```

## Brooks-Corey Hysteresis Parameters

Fitted to experimental H₂-brine data (Boon et al., 2022):

| Parameter | Drainage | Imbibition |
|-----------|----------|------------|
| S_wc | 0.45 | 0.45 |
| S_gr | 0.02 | 0.24 |
| n_w | 4.01 | 4.00 |
| n_g | 4.66 | 2.80 |
| k_rw^max | 0.95 | 0.90 |
| k_rg^max | 0.20 | 0.17 |

## Operational Sequence

Each simulation models a **2-cycle UHS operation**:

1. **Primary Injection**: τ days (e.g., 180 days)
2. **Primary Withdrawal**: τ/2 days (90 days)
3. **Secondary Injection**: τ/2 days (90 days)
4. **Secondary Withdrawal**: τ/2 days (90 days)

**Total**: ~450 days (15 months), 34 timesteps at 13.2-day intervals

## Computational Requirements

### For Full Dataset Generation:
- **CPU**: Multi-core recommended (parallel simulations)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 60 GB free space (50 GB data + logs)
- **Time**: ~1 week on a modern workstation

### Parallel Execution (Optional)

To speed up, you can modify the script to use MATLAB's parallel pool:

```matlab
parpool(4)  % Use 4 cores
parfor i = 1:num_simulations
    % Run simulations in parallel
end
```

## Troubleshooting

### "MRST not found" Error
```matlab
% Add MRST to path
addpath('/path/to/mrst')
startup
```

### "Out of memory" Error
- Reduce number of simultaneous simulations
- Increase MATLAB heap size
- Run in smaller batches

### Slow Simulation
- Expected: 3-15 min per simulation
- Check grid resolution settings
- Ensure no other heavy processes running

## Data Availability

Due to GitHub file size limits (max 100 MB per file), the **complete dataset is hosted separately**:

### Option 1: Download Pre-generated Data
- **Zenodo**: [Link to be added after upload]
- **Google Drive**: [Link to be added]
- **Size**: ~50 GB (compressed)

### Option 2: Generate Your Own Data
- Run `generate_multifidelity_data.m`
- Estimated time: 1 week
- Advantage: Full control over parameters

## File Structure After Generation

```
multifidelity_deeponet_data/
├── low_fidelity/
│   ├── simulation_00001.mat
│   ├── simulation_00002.mat
│   └── ... (2000 files)
├── medium_fidelity/
│   ├── simulation_00001.mat
│   └── ... (500 files)
├── high_fidelity/
│   ├── simulation_00001.mat
│   └── ... (100 files)
├── logs/
│   ├── master_log_YYYYMMDD_HHMMSS.txt
│   └── google_drive_sync_status.txt
└── backups/
```

## Citation

If you use these scripts or the generated data, please cite:

```bibtex
@article{kabbaj2025uhs,
  title={Simulation-Based Staged Decoupling of Mobility, Heterogeneity and Hysteresis Effects in Underground Hydrogen Storage},
  author={Kabbaj, Narjisse},
  journal={Geoenergy Science and Engineering},
  year={2025}
}
```

## Contact

For questions about data generation or MRST setup:

**Narjisse Kabbaj**
Email: nkabbaj@effatuniversity.edu.sa

---

**Note**: The actual data files are NOT included in this GitHub repository due to size constraints. Please download from the links provided above or generate your own data using these scripts.
