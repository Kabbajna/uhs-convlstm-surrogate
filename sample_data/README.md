# Sample Data

This folder contains **2 sample simulations** from the medium-fidelity dataset for testing and demonstration purposes.

## ⚠️ This is NOT the complete dataset!

**Full dataset size**: 6.4 GB (2,600 simulations)
- Low fidelity: 73 MB (2000 simulations)
- Medium fidelity: 712 MB (500 simulations)
- High fidelity: 5.6 GB (100 simulations)

## Download Full Dataset

The complete dataset is available at:

### Option 1: Zenodo (Recommended - with DOI)
🔗 **[TO BE ADDED]** - Upload your data to Zenodo and add link here

### Option 2: Google Drive
🔗 **[TO BE ADDED]** - Share your Google Drive link here

### Option 3: Figshare
🔗 **[TO BE ADDED]** - Alternative research data repository

## How to Upload Your Data (For You, Narjisse)

### Upload to Zenodo (FREE, gets you a DOI):

1. Go to: https://zenodo.org/
2. Sign in (use your GitHub account)
3. Click "Upload" → "New upload"
4. Upload your data folders (can be zipped)
5. Fill in metadata:
   - Title: "Multi-Fidelity UHS Simulation Dataset"
   - Authors: Narjisse Kabbaj
   - Description: Copy from DATA_README.md
   - Keywords: hydrogen storage, reservoir simulation, MRST, multi-fidelity
6. Publish → You get a DOI!
7. Add the DOI link here

### Or Upload to Google Drive:

1. Upload your `medium_fidelity/` and `high_fidelity/` folders
2. Right-click → "Get link" → "Anyone with the link"
3. Copy the link
4. Add it here

## Sample Data Contents

This folder contains:
- `medium_fidelity/simulation_00001.mat` - First simulation
- `medium_fidelity/simulation_00002.mat` - Second simulation

Each file contains 34 timesteps (15 months) with:
- Water saturation (Sw)
- Gas saturation (Sg)
- Pressure (P)
- Historical max gas saturation (Sg_max)
- Reservoir parameters (K, φ)

## Using the Sample Data

```python
import scipy.io as sio

# Load sample simulation
data = sio.loadmat('sample_data/medium_fidelity/simulation_00001.mat')
sim = data['sim_data']

# Access timesteps
timesteps = sim['timesteps'][0, 0]
first_timestep = timesteps[0]

# Get saturation fields (20×20×20)
Sw = first_timestep['Sw']
Sg = first_timestep['Sg']
P = first_timestep['P']

print(f"Grid shape: {Sw.shape}")
print(f"Number of timesteps: {len(timesteps)}")
```

## File Format

See main `DATA_README.md` for detailed data format specifications.

---

**For the complete dataset (6.4 GB), please use the download links above.**
