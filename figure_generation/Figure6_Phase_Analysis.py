"""
===============================================================================
FIGURE 6: Phase-wise Performance Analysis (v2 - Fixed overlaps)
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

PHASE_COLORS = {
    'injection': '#1565C0',
    'storage': '#546E7A', 
    'production': '#C62828',
}

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


def generate_rollout(model, sim_data, device, n_steps=11):
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
    
    r2_per_step = []
    
    with torch.no_grad():
        for t in range(min(n_steps, len(timesteps) - 1)):
            pred, states = model(current_input, states)
            pred_sg = pred[0, 0].cpu().numpy()
            gt_sg = timesteps[t + 1].Sg
            
            r2 = 1 - np.sum((gt_sg - pred_sg)**2) / np.sum((gt_sg - np.mean(gt_sg))**2)
            r2_per_step.append(r2)
            
            Sg_pred, Sw_pred, P_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
            Sg_max_gt = torch.tensor(timesteps[t].Sg_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            Sg_max_updated = torch.maximum(Sg_max_gt, Sg_pred)
            current_input = torch.cat([Sw_pred, Sg_pred, P_pred, Sg_max_updated, meta], dim=1)
    
    return r2_per_step


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GENERATING FIGURE 6: Phase-wise Performance Analysis")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model...")
    model = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=BASE_FEATURES).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Collect R² from multiple simulations
    print(f"\n🔄 Evaluating {N_SIMS} simulations...")
    
    data_dir = DATA_PATH / 'medium_fidelity'
    mat_files = sorted(list(data_dir.glob("*.mat")))[:N_SIMS]
    
    all_r2_curves = []
    
    for i, mat_file in enumerate(mat_files):
        sim_data = load_mat_flexible(mat_file)
        if sim_data is None or len(sim_data['timesteps']) < 12:
            continue
        
        r2_per_step = generate_rollout(model, sim_data, device, n_steps=11)
        if len(r2_per_step) == 11:
            all_r2_curves.append(r2_per_step)
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i+1}/{N_SIMS} simulations...")
    
    print(f"✅ Collected {len(all_r2_curves)} valid simulations")
    
    # Convert to numpy and compute stats
    r2_array = np.array(all_r2_curves)
    r2_mean = np.mean(r2_array, axis=0)
    r2_std = np.std(r2_array, axis=0)
    
    months = np.arange(1, 12)
    
    # Phase averages
    r2_injection = np.mean(r2_mean[0:4])
    r2_storage = np.mean(r2_mean[4:7])
    r2_production = np.mean(r2_mean[7:11])
    
    print(f"\n📊 Phase-wise R²:")
    print(f"   Injection (M1-4):   {r2_injection:.4f}")
    print(f"   Storage (M5-7):     {r2_storage:.4f}")
    print(f"   Production (M8-11): {r2_production:.4f}")
    
    # ==================== CREATE FIGURE ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: R² evolution by timestep =====
    
    # Phase background regions
    ax1.axvspan(0.5, 4.5, alpha=0.15, color=PHASE_COLORS['injection'])
    ax1.axvspan(4.5, 7.5, alpha=0.15, color=PHASE_COLORS['storage'])
    ax1.axvspan(7.5, 11.5, alpha=0.15, color=PHASE_COLORS['production'])
    
    # Plot mean with confidence band
    ax1.fill_between(months, r2_mean - r2_std, r2_mean + r2_std, 
                     alpha=0.3, color='#1976D2', label='±1 std')
    ax1.plot(months, r2_mean, 'o-', color='#1976D2', linewidth=2, markersize=8, 
             markerfacecolor='white', markeredgewidth=2, label='Mean $R^2$')
    
    # Phase labels INSIDE the plot area (at y=0.53)
    ax1.text(2.5, 0.53, 'Injection', ha='center', fontsize=10, fontweight='bold', 
            color=PHASE_COLORS['injection'])
    ax1.text(6, 0.53, 'Storage', ha='center', fontsize=10, fontweight='bold',
            color=PHASE_COLORS['storage'])
    ax1.text(9.5, 0.53, 'Production', ha='center', fontsize=10, fontweight='bold',
            color=PHASE_COLORS['production'])
    
    # Reference line
    ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(11.3, 0.95, '$R^2$=0.95', fontsize=9, color='green', va='center')
    
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('$R^2$ Score', fontsize=12)
    ax1.set_xlim(0.5, 11.5)
    ax1.set_ylim(0.45, 1.02)
    ax1.set_xticks(months)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) $R^2$ Evolution Across Operational Cycle', fontsize=12, fontweight='bold', pad=12)
    
    # ===== RIGHT PLOT: Bar chart by phase =====
    
    phases = ['Injection\n(M1-4)', 'Storage\n(M5-7)', 'Production\n(M8-11)']
    phase_r2 = [r2_injection, r2_storage, r2_production]
    phase_std = [np.std(r2_mean[0:4]), np.std(r2_mean[4:7]), np.std(r2_mean[7:11])]
    colors = [PHASE_COLORS['injection'], PHASE_COLORS['storage'], PHASE_COLORS['production']]
    
    bars = ax2.bar(phases, phase_r2, color=colors, edgecolor='white', linewidth=2, width=0.6)
    ax2.errorbar(phases, phase_r2, yerr=phase_std, fmt='none', color='black', capsize=5, capthick=2)
    
    # Value labels on bars
    for bar, r2 in zip(bars, phase_r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'{r2:.3f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    # Highlight best phase
    best_idx = np.argmax(phase_r2)
    ax2.plot(best_idx, phase_r2[best_idx] + 0.03, marker='*', markersize=15, 
            color='#4CAF50', markeredgecolor='white', markeredgewidth=0.5)
    
    ax2.set_ylabel('Mean $R^2$ Score', fontsize=12)
    ax2.set_ylim(0.45, 1.05)
    ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title('(b) Average $R^2$ by Operational Phase', fontsize=12, fontweight='bold', pad=12)
    
    # Clean spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # ==================== SAVE ====================
    output_png = OUTPUT_DIR / 'Figure6_Phase_Analysis.png'
    output_pdf = OUTPUT_DIR / 'Figure6_Phase_Analysis.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close()
