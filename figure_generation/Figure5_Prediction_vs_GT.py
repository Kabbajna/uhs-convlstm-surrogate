"""
===============================================================================
FIGURE 5 FINAL: Prediction vs Ground Truth
===============================================================================
Using medium_sim_0020.mat - Clean publication version
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
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
SIM_FILE = 'medium_sim_0020.mat'

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
# MODEL ARCHITECTURE
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
    
    predictions, ground_truths, r2_per_step = [], [], []
    
    with torch.no_grad():
        for t in range(min(n_steps, len(timesteps) - 1)):
            pred, states = model(current_input, states)
            pred_sg = pred[0, 0].cpu().numpy()
            gt_sg = timesteps[t + 1].Sg
            
            predictions.append(pred_sg)
            ground_truths.append(gt_sg)
            r2 = 1 - np.sum((gt_sg - pred_sg)**2) / np.sum((gt_sg - np.mean(gt_sg))**2)
            r2_per_step.append(r2)
            
            Sg_pred, Sw_pred, P_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
            Sg_max_gt = torch.tensor(timesteps[t].Sg_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            Sg_max_updated = torch.maximum(Sg_max_gt, Sg_pred)
            current_input = torch.cat([Sw_pred, Sg_pred, P_pred, Sg_max_updated, meta], dim=1)
    
    return predictions, ground_truths, r2_per_step


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GENERATING FIGURE 5 FINAL")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model...")
    model = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=BASE_FEATURES).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load simulation
    print(f"📂 Loading {SIM_FILE}...")
    sim_data = load_mat_flexible(DATA_PATH / 'medium_fidelity' / SIM_FILE)
    timesteps = sim_data['timesteps']
    nx, ny, nz = timesteps[0].Sg.shape
    y_slice = ny // 2
    
    # Generate predictions
    print("🔄 Generating predictions...")
    predictions, ground_truths, r2_per_step = generate_rollout(model, sim_data, device, n_steps=11)
    
    # Overall R²
    all_pred = np.concatenate([p.flatten() for p in predictions])
    all_gt = np.concatenate([g.flatten() for g in ground_truths])
    overall_r2 = 1 - np.sum((all_gt - all_pred)**2) / np.sum((all_gt - np.mean(all_gt))**2)
    
    print(f"\n📊 Results:")
    print(f"   R² Month 3:  {r2_per_step[2]:.3f}")
    print(f"   R² Month 6:  {r2_per_step[5]:.3f}")
    print(f"   R² Month 10: {r2_per_step[9]:.3f}")
    print(f"   Overall:     {overall_r2:.3f}")
    
    # Create figure
    timestep_indices = [2, 5, 9]
    phase_names = ['Injection (Month 3)', 'Storage (Month 6)', 'Production (Month 10)']
    phase_colors = ['#1565C0', '#546E7A', '#C62828']
    
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.15,
                          left=0.08, right=0.88, top=0.88, bottom=0.08)
    
    all_data = predictions + ground_truths
    vmax = min(0.5, np.ceil(max([d.max() for d in all_data[:11]]) * 10) / 10 + 0.05)
    
    for col, (t_idx, phase_name, color) in enumerate(zip(timestep_indices, phase_names, phase_colors)):
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            data = ground_truths[t_idx] if row == 0 else predictions[t_idx]
            row_label = 'Ground Truth' if row == 0 else 'Prediction'
            
            slice_data = data[:, y_slice, :]
            ax.imshow(slice_data.T, cmap='viridis', vmin=0, vmax=vmax,
                     aspect='auto', origin='upper', extent=[0, nx, nz, 0])
            
            if row == 1:
                ax.set_xlabel('X (cells)', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if col == 0:
                ax.set_ylabel(f'{row_label}\nDepth Z', fontsize=10)
            
            if row == 0:
                ax.set_title(phase_name, fontsize=11, fontweight='bold', color=color, pad=8)
            
            if row == 1:
                mae = np.mean(np.abs(ground_truths[t_idx] - predictions[t_idx]))
                r2 = r2_per_step[t_idx]
                ax.text(0.03, 0.97, f'MAE={mae:.4f}\n$R^2$={r2:.3f}',
                       transform=ax.transAxes, fontsize=8, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9))
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color if row == 0 else 'gray')
                spine.set_linewidth(1.5 if row == 0 else 1)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'Gas Saturation $S_g$', fontsize=11)
    
    # Clean title
    fig.suptitle('ConvLSTM-UNet: Prediction vs Ground Truth (Autoregressive Rollout)',
                fontsize=13, fontweight='bold', y=0.96)
    
    # Save
    output_png = OUTPUT_DIR / 'Figure5_Prediction_vs_GT.png'
    output_pdf = OUTPUT_DIR / 'Figure5_Prediction_vs_GT.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    plt.close()
