"""
===============================================================================
FIGURE 7: Spatial Error Distribution Analysis (v2 - 2 panels)
===============================================================================
(a) Error heatmap (XZ slice)
(b) Radial error profile (error vs distance from well)
===============================================================================
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path('/Users/narjisse/Documents/Effat Courses/deeponet')
OUTPUT_DIR = DATA_PATH / 'publication_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = DATA_PATH / 'checkpoints' / 'best_convlstm_SS_70_15_15.pt'
BASE_FEATURES = 32
N_SIMS = 20

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'dejavuserif',
})

device = torch.device('cpu')

# ============================================================================
# MODEL ARCHITECTURE (compact)
# ============================================================================

class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv3d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        nx, ny, nz = spatial_size
        return (torch.zeros(batch_size, self.hidden_channels, nx, ny, nz, device=device),
                torch.zeros(batch_size, self.hidden_channels, nx, ny, nz, device=device))


class ConvLSTMEncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convlstm = ConvLSTMCell3D(in_channels, hidden_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x, h_prev, c_prev):
        h, c = self.convlstm(x, h_prev, c_prev)
        return self.pool(h), h, h, c


class ConvLSTMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convlstm = ConvLSTMCell3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, h_prev, c_prev):
        x = self.upconv(x)
        if x.shape != skip.shape:
            d, h, w = skip.size(2) - x.size(2), skip.size(3) - x.size(3), skip.size(4) - x.size(4)
            x = F.pad(x, [w//2, w-w//2, h//2, h-h//2, d//2, d-d//2])
        combined = torch.cat([x, skip], dim=1)
        h, c = self.convlstm(combined, h_prev, c_prev)
        return h, h, c


class ConvLSTMUNet3D(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_features=32):
        super().__init__()
        self.enc1 = ConvLSTMEncoderBlock(in_channels, base_features)
        self.enc2 = ConvLSTMEncoderBlock(base_features, base_features*2)
        self.enc3 = ConvLSTMEncoderBlock(base_features*2, base_features*4)
        self.bottleneck = ConvLSTMCell3D(base_features*4, base_features*8)
        self.dec3 = ConvLSTMDecoderBlock(base_features*8, base_features*4, base_features*4)
        self.dec2 = ConvLSTMDecoderBlock(base_features*4, base_features*2, base_features*2)
        self.dec1 = ConvLSTMDecoderBlock(base_features*2, base_features, base_features)
        self.output = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden_states(self, batch_size, spatial_sizes, device):
        states = {}
        states['enc1_h'], states['enc1_c'] = self.enc1.convlstm.init_hidden(batch_size, spatial_sizes[0], device)
        states['enc2_h'], states['enc2_c'] = self.enc2.convlstm.init_hidden(batch_size, spatial_sizes[1], device)
        states['enc3_h'], states['enc3_c'] = self.enc3.convlstm.init_hidden(batch_size, spatial_sizes[2], device)
        states['bn_h'], states['bn_c'] = self.bottleneck.init_hidden(batch_size, spatial_sizes[3], device)
        states['dec3_h'], states['dec3_c'] = self.dec3.convlstm.init_hidden(batch_size, spatial_sizes[2], device)
        states['dec2_h'], states['dec2_c'] = self.dec2.convlstm.init_hidden(batch_size, spatial_sizes[1], device)
        states['dec1_h'], states['dec1_c'] = self.dec1.convlstm.init_hidden(batch_size, spatial_sizes[0], device)
        return states

    def forward(self, x, states):
        x1, skip1, states['enc1_h'], states['enc1_c'] = self.enc1(x, states['enc1_h'], states['enc1_c'])
        x2, skip2, states['enc2_h'], states['enc2_c'] = self.enc2(x1, states['enc2_h'], states['enc2_c'])
        x3, skip3, states['enc3_h'], states['enc3_c'] = self.enc3(x2, states['enc3_h'], states['enc3_c'])
        states['bn_h'], states['bn_c'] = self.bottleneck(x3, states['bn_h'], states['bn_c'])
        x, states['dec3_h'], states['dec3_c'] = self.dec3(states['bn_h'], skip3, states['dec3_h'], states['dec3_c'])
        x, states['dec2_h'], states['dec2_c'] = self.dec2(x, skip2, states['dec2_h'], states['dec2_c'])
        x, states['dec1_h'], states['dec1_c'] = self.dec1(x, skip1, states['dec1_h'], states['dec1_c'])
        return self.sigmoid(self.output(x)), states


def get_spatial_sizes(nx, ny, nz):
    return [(nx, ny, nz), (nx//2, ny//2, nz//2), (nx//4, ny//4, nz//4), (nx//8, ny//8, nz//8)]


# ============================================================================
# DATA & PREDICTION
# ============================================================================

def load_mat_flexible(mat_path):
    try:
        data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        sim = data['sim_data']
        if isinstance(sim, np.ndarray) and sim.ndim == 2:
            sim = sim[0, 0]
        
        timesteps_raw = sim.timesteps.flatten() if hasattr(sim.timesteps, 'flatten') else sim.timesteps
        if not isinstance(timesteps_raw, np.ndarray):
            timesteps_raw = [timesteps_raw]
        
        params = sim.params if hasattr(sim, 'params') else None
        if params is not None and isinstance(params, np.ndarray) and params.ndim == 2:
            params = params[0, 0]
        K = float(params.K[0, 0]) if params and hasattr(params.K, 'shape') else 1e-13
        
        timesteps = []
        for ts in timesteps_raw:
            timesteps.append(type('obj', (object,), {
                'Sw': np.array(ts.Sw, dtype=np.float32),
                'Sg': np.array(ts.Sg, dtype=np.float32),
                'P': np.array(ts.P, dtype=np.float32),
                'Sg_max': np.array(ts.Sg_max, dtype=np.float32),
            })())
        return {'timesteps': timesteps, 'K': K}
    except:
        return None


def generate_rollout_with_errors(model, sim_data, device, n_steps=11):
    model.eval()
    timesteps = sim_data['timesteps']
    K = sim_data['K']
    first_ts = timesteps[0]
    nx, ny, nz = first_ts.Sg.shape
    
    spatial_sizes = get_spatial_sizes(nx, ny, nz)
    states = model.init_hidden_states(1, spatial_sizes, device)
    
    Sw = torch.tensor(first_ts.Sw, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    Sg = torch.tensor(first_ts.Sg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    P = torch.tensor(first_ts.P, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 14.8e6
    Sg_max = torch.tensor(first_ts.Sg_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    meta = torch.ones(1, 1, nx, ny, nz, device=device) * (K / 1e-13)
    
    current_input = torch.cat([Sw, Sg, P, Sg_max, meta], dim=1)
    
    spatial_errors = np.zeros((nx, ny, nz), dtype=np.float32)
    
    with torch.no_grad():
        for t in range(min(n_steps, len(timesteps) - 1)):
            pred, states = model(current_input, states)
            pred_sg = pred[0, 0].cpu().numpy()
            gt_sg = timesteps[t + 1].Sg
            
            spatial_errors += np.abs(gt_sg - pred_sg)
            
            Sg_pred, Sw_pred, P_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
            Sg_max_gt = torch.tensor(timesteps[t].Sg_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            Sg_max_updated = torch.maximum(Sg_max_gt, Sg_pred)
            current_input = torch.cat([Sw_pred, Sg_pred, P_pred, Sg_max_updated, meta], dim=1)
    
    spatial_errors /= n_steps
    return spatial_errors


def compute_radial_profile(error_3d, well_center=(10, 10, 0)):
    nx, ny, nz = error_3d.shape
    well_x, well_y, well_z = well_center
    
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    dist = np.sqrt((xx - well_x)**2 + (yy - well_y)**2)
    error_2d = np.mean(error_3d, axis=2)
    
    max_dist = np.sqrt(2) * max(nx, ny) / 2
    n_bins = 10
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_errors = []
    std_errors = []
    
    for i in range(n_bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i+1])
        if np.sum(mask) > 0:
            mean_errors.append(np.mean(error_2d[mask]))
            std_errors.append(np.std(error_2d[mask]))
        else:
            mean_errors.append(np.nan)
            std_errors.append(np.nan)
    
    return bin_centers, np.array(mean_errors), np.array(std_errors)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GENERATING FIGURE 7: Spatial Error Distribution (2 panels)")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model...")
    model = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=BASE_FEATURES).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Collect spatial errors
    print(f"\n🔄 Evaluating {N_SIMS} simulations...")
    
    data_dir = DATA_PATH / 'medium_fidelity'
    mat_files = sorted(list(data_dir.glob("*.mat")))[:N_SIMS]
    
    all_spatial_errors = []
    nx, ny, nz = None, None, None
    
    for i, mat_file in enumerate(mat_files):
        sim_data = load_mat_flexible(mat_file)
        if sim_data is None or len(sim_data['timesteps']) < 12:
            continue
        
        spatial_errors = generate_rollout_with_errors(model, sim_data, device, n_steps=11)
        all_spatial_errors.append(spatial_errors)
        
        if nx is None:
            nx, ny, nz = spatial_errors.shape
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i+1}/{N_SIMS} simulations...")
    
    print(f"✅ Collected {len(all_spatial_errors)} valid simulations")
    
    mean_spatial_error = np.mean(all_spatial_errors, axis=0)
    well_center = (nx // 2, ny // 2, 0)
    radial_dist, radial_error_mean, radial_error_std = compute_radial_profile(mean_spatial_error, well_center)
    
    print(f"\n📊 Spatial Error Statistics:")
    print(f"   Max MAE: {np.max(mean_spatial_error):.5f}")
    print(f"   Mean MAE: {np.mean(mean_spatial_error):.5f}")
    
    # ==================== CREATE FIGURE (2 panels) ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    y_slice = ny // 2
    
    # ===== (a) Error heatmap - XZ slice =====
    error_slice_xz = mean_spatial_error[:, y_slice, :].T
    
    im1 = ax1.imshow(error_slice_xz, cmap='hot', aspect='auto', origin='upper',
                    extent=[0, nx, nz, 0])
    
    ax1.set_xlabel('X (cells)', fontsize=12)
    ax1.set_ylabel('Depth Z (cells)', fontsize=12)
    ax1.set_title('(a) MAE Spatial Distribution (Y=10 slice)', fontsize=12, fontweight='bold', pad=12)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cbar1.set_label('Mean Absolute Error', fontsize=11)
    
    # ===== (b) Radial error profile =====
    valid = ~np.isnan(radial_error_mean)
    
    # Background zones
    ax2.axvspan(0, 3, alpha=0.2, color='#C62828', label='Near-well zone')
    ax2.axvspan(3, 14, alpha=0.1, color='#546E7A', label='Far-field zone')
    
    # Plot with error band
    ax2.fill_between(radial_dist[valid], 
                     (radial_error_mean - radial_error_std)[valid],
                     (radial_error_mean + radial_error_std)[valid],
                     alpha=0.3, color='#1976D2')
    ax2.plot(radial_dist[valid], radial_error_mean[valid], 'o-', color='#1976D2', 
            linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2,
            label='Mean MAE ± std')
    
    # Annotations
    near_well_mae = np.mean(radial_error_mean[radial_dist < 3])
    far_field_mae = np.mean(radial_error_mean[radial_dist >= 3])
    
    ax2.annotate(f'Near-well\nMAE≈{near_well_mae:.4f}', 
                xy=(1.5, near_well_mae), xytext=(2, 0.018),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#C62828', alpha=0.9))
    
    ax2.annotate(f'Far-field\nMAE≈{far_field_mae:.4f}', 
                xy=(8, far_field_mae), xytext=(9, 0.008),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#546E7A', alpha=0.9))
    
    ax2.set_xlabel('Distance from Well (cells)', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax2.set_title('(b) Radial Error Profile', fontsize=12, fontweight='bold', pad=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(radial_dist[valid]) + 0.5)
    ax2.set_ylim(0, max(radial_error_mean[valid]) * 1.3)
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # ==================== SAVE ====================
    output_png = OUTPUT_DIR / 'Figure7_Spatial_Error.png'
    output_pdf = OUTPUT_DIR / 'Figure7_Spatial_Error.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close()
