"""
ConvLSTM U-Net avec Scheduled Sampling pour UHS
VERSION MAC LOCAL - SEED=42 - SPLIT 70/15/15
MÊME SCHEDULE QUE L'ORIGINAL (inverse_sigmoid k=5 + Pure Rollout epochs 31-40)
"""

# ============================================================================
# SETUP MAC LOCAL
# ============================================================================

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIXER TOUS LES SEEDS POUR REPRODUCTIBILITÉ
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"🎲 SEED = {SEED} (résultats reproductibles)")

# ============================================================================
# CHEMINS MAC LOCAL
# ============================================================================
DATA_PATH = Path('/Users/narjisse/Documents/Effat Courses/deeponet')
CHECKPOINT_DIR = DATA_PATH / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DÉTECTION MATÉRIEL MAC
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("cpu")  # MPS ne supporte pas ConvTranspose3d
    print("⚠️  Apple Silicon détecté mais MPS non utilisable")
    print("   Raison: ConvTranspose3d non supporté sur MPS")
    print("   Utilisation: CPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA GPU détecté: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  CPU mode")

print(f"\nDevice: {device}")
print(f"Data path: {DATA_PATH}")
print(f"Data exists: {DATA_PATH.exists()}")

# Vérifier les dossiers de données
for fidelity in ['low_fidelity', 'medium_fidelity', 'high_fidelity']:
    fidelity_path = DATA_PATH / fidelity
    if fidelity_path.exists():
        n_files = len(list(fidelity_path.glob('*.mat')))
        print(f"  ✅ {fidelity}: {n_files} fichiers .mat")
    else:
        print(f"  ❌ {fidelity}: DOSSIER MANQUANT")

print(f"\nCheckpoints: {CHECKPOINT_DIR}")
print("="*70 + "\n")

# ============================================================================
# SCHEDULED SAMPLING CLASS - EXACTEMENT COMME L'ORIGINAL
# ============================================================================

class ScheduledSampling:
    """Gère la décroissance progressive du teacher forcing"""
    
    def __init__(self, schedule_type='inverse_sigmoid', k=5):
        self.schedule_type = schedule_type
        self.k = k
    
    def get_sampling_prob(self, epoch, max_epochs):
        if self.schedule_type == 'linear':
            return max(0.1, 1.0 - (epoch / max_epochs) * 0.9)
        elif self.schedule_type == 'exponential':
            return max(0.1, 0.9 ** epoch)
        elif self.schedule_type == 'inverse_sigmoid':
            return max(0.1, self.k / (self.k + np.exp(epoch / self.k)))
        return 1.0

# ============================================================================
# ConvLSTM CELL
# ============================================================================

class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        nx, ny, nz = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, nx, ny, nz, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, nx, ny, nz, device=device)
        return h, c

# ============================================================================
# ConvLSTM ENCODER/DECODER
# ============================================================================

class ConvLSTMEncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convlstm = ConvLSTMCell3D(in_channels, hidden_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x, h_prev, c_prev):
        h, c = self.convlstm(x, h_prev, c_prev)
        skip = h
        pooled = self.pool(h)
        return pooled, skip, h, c


class ConvLSTMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convlstm = ConvLSTMCell3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, h_prev, c_prev):
        x = self.upconv(x)
        if x.shape != skip.shape:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [diff_w//2, diff_w-diff_w//2,
                         diff_h//2, diff_h-diff_h//2,
                         diff_d//2, diff_d-diff_d//2])
        combined = torch.cat([x, skip], dim=1)
        h, c = self.convlstm(combined, h_prev, c_prev)
        return h, h, c

# ============================================================================
# ConvLSTM U-NET
# ============================================================================

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
        self.base_features = base_features

    def init_hidden_states(self, batch_size, spatial_sizes):
        states = {}
        states['enc1_h'], states['enc1_c'] = self.enc1.convlstm.init_hidden(batch_size, spatial_sizes[0])
        states['enc2_h'], states['enc2_c'] = self.enc2.convlstm.init_hidden(batch_size, spatial_sizes[1])
        states['enc3_h'], states['enc3_c'] = self.enc3.convlstm.init_hidden(batch_size, spatial_sizes[2])
        states['bn_h'], states['bn_c'] = self.bottleneck.init_hidden(batch_size, spatial_sizes[3])
        states['dec3_h'], states['dec3_c'] = self.dec3.convlstm.init_hidden(batch_size, spatial_sizes[2])
        states['dec2_h'], states['dec2_c'] = self.dec2.convlstm.init_hidden(batch_size, spatial_sizes[1])
        states['dec1_h'], states['dec1_c'] = self.dec1.convlstm.init_hidden(batch_size, spatial_sizes[0])
        return states

    def forward(self, x, states):
        x1, skip1, states['enc1_h'], states['enc1_c'] = self.enc1(x, states['enc1_h'], states['enc1_c'])
        x2, skip2, states['enc2_h'], states['enc2_c'] = self.enc2(x1, states['enc2_h'], states['enc2_c'])
        x3, skip3, states['enc3_h'], states['enc3_c'] = self.enc3(x2, states['enc3_h'], states['enc3_c'])
        states['bn_h'], states['bn_c'] = self.bottleneck(x3, states['bn_h'], states['bn_c'])
        x, states['dec3_h'], states['dec3_c'] = self.dec3(states['bn_h'], skip3, states['dec3_h'], states['dec3_c'])
        x, states['dec2_h'], states['dec2_c'] = self.dec2(x, skip2, states['dec2_h'], states['dec2_c'])
        x, states['dec1_h'], states['dec1_c'] = self.dec1(x, skip1, states['dec1_h'], states['dec1_c'])
        x = self.output(x)
        x = self.sigmoid(x)
        return x, states

# ============================================================================
# DATASET
# ============================================================================

class SequenceUHSDataset(Dataset):
    def __init__(self, data_path, fidelity='medium', sequence_length=10):
        self.data_dir = data_path / f"{fidelity}_fidelity"
        self.sequence_length = sequence_length
        self.fidelity = fidelity
        self.mat_files = sorted(list(self.data_dir.glob("*.mat")))
        self.sequences = self._create_sequences()
        print(f"Dataset {fidelity}: {len(self.mat_files)} sims, {len(self.sequences)} sequences")

    def _load_mat_flexible(self, mat_path):
        try:
            data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            if 'sim_data' not in data:
                return None
            sim = data['sim_data']
            if isinstance(sim, np.ndarray) and sim.ndim == 2:
                sim = sim[0, 0]
            if not hasattr(sim, 'timesteps'):
                return None
            timesteps_raw = sim.timesteps.flatten() if hasattr(sim.timesteps, 'flatten') else sim.timesteps
            if not isinstance(timesteps_raw, np.ndarray):
                timesteps_raw = [timesteps_raw]
            if hasattr(sim, 'params'):
                params = sim.params
                if isinstance(params, np.ndarray) and params.ndim == 2:
                    params = params[0, 0]
                K = float(params.K[0, 0]) if hasattr(params.K, 'shape') else float(params.K)
                phi = float(params.phi[0, 0]) if hasattr(params.phi, 'shape') else float(params.phi)
            else:
                K = 1e-13
                phi = 0.2
            timesteps = []
            for ts in timesteps_raw:
                def safe_extract(field, default=0):
                    val = getattr(ts, field, default)
                    if hasattr(val, 'shape') and val.ndim >= 1:
                        return val[0, 0] if val.ndim == 2 else val[0]
                    return val
                timestep_obj = type('obj', (object,), {
                    'Sw': np.array(ts.Sw, dtype=np.float32),
                    'Sg': np.array(ts.Sg, dtype=np.float32),
                    'P': np.array(ts.P, dtype=np.float32),
                    'Sg_max': np.array(ts.Sg_max, dtype=np.float32),
                    'time': float(safe_extract('time', 0)),
                    'Q': float(safe_extract('Q', 0)),
                    'operation': str(safe_extract('operation', 'injection'))
                })()
                timesteps.append(timestep_obj)
            return {'timesteps': timesteps, 'K': K, 'phi': phi}
        except Exception as e:
            return None

    def _create_sequences(self):
        sequences = []
        error_count = 0
        for mat_file in self.mat_files:
            result = self._load_mat_flexible(mat_file)
            if result is None:
                error_count += 1
                continue
            timesteps = result['timesteps']
            if len(timesteps) < self.sequence_length + 1:
                continue
            for i in range(len(timesteps) - self.sequence_length):
                sequences.append({
                    'file': mat_file,
                    'start_idx': i,
                    'K': result['K'],
                    'phi': result['phi']
                })
        if error_count > 0:
            print(f"Skipped {error_count} files")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        result = self._load_mat_flexible(seq_info['file'])
        if result is None:
            raise ValueError(f"Cannot load {seq_info['file']}")
        timesteps = result['timesteps']
        start = seq_info['start_idx']
        sequence = timesteps[start:start + self.sequence_length + 1]
        inputs = []
        targets = []
        for i in range(self.sequence_length):
            current = sequence[i]
            next_step = sequence[i + 1]
            Sw = np.array(current.Sw, dtype=np.float32)
            Sg = np.array(current.Sg, dtype=np.float32)
            P = np.array(current.P, dtype=np.float32) / 14.8e6
            Sg_max = np.array(current.Sg_max, dtype=np.float32)
            meta = np.ones_like(Sw) * (seq_info['K'] / 1e-13)
            input_state = np.stack([Sw, Sg, P, Sg_max, meta], axis=0)
            target_state = np.stack([
                np.array(next_step.Sg, dtype=np.float32),
                np.array(next_step.Sw, dtype=np.float32),
                np.array(next_step.P, dtype=np.float32) / 14.8e6
            ], axis=0)
            inputs.append(input_state)
            targets.append(target_state)
        return {
            'inputs': np.array(inputs),
            'targets': np.array(targets),
            'K': seq_info['K'],
            'phi': seq_info['phi']
        }

# ============================================================================
# LOSS PHYSICS-INFORMED
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_sat=0.0, lambda_mass=0.0, lambda_darcy=0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_sat = lambda_sat
        self.lambda_mass = lambda_mass
        self.lambda_darcy = lambda_darcy

    def forward(self, pred, target, input_state=None, metadata=None):
        loss_data = F.mse_loss(pred, target)
        Sg_pred = pred[:, 0]
        Sw_pred = pred[:, 1]
        loss_sat = torch.mean((Sg_pred + Sw_pred - 1.0) ** 2)
        loss_bounds = torch.mean(F.relu(-Sg_pred)) + torch.mean(F.relu(Sg_pred - 1.0))
        loss_bounds += torch.mean(F.relu(-Sw_pred)) + torch.mean(F.relu(Sw_pred - 1.0))
        loss_mass = torch.tensor(0.0, device=pred.device)
        if input_state is not None and metadata is not None:
            phi = metadata.get('phi', 0.2)
            dt = metadata.get('dt', 1.0)
            Sg_t = input_state[:, 1]
            dSg_dt = (Sg_pred - Sg_t) / (dt + 1e-8)
            operation = metadata.get('operation', 0.5)
            if operation > 0.5:
                loss_mass_dir = torch.mean(F.relu(-dSg_dt))
            else:
                loss_mass_dir = torch.mean(F.relu(dSg_dt))
            loss_mass_rate = torch.mean(torch.abs(phi * dSg_dt) ** 2) * 0.01
            loss_mass = loss_mass_dir + loss_mass_rate
        loss_darcy = torch.tensor(0.0, device=pred.device)
        if input_state is not None and metadata is not None:
            Sg_grad_x = Sg_pred[:, 1:, :, :] - Sg_pred[:, :-1, :, :]
            Sg_grad_y = Sg_pred[:, :, 1:, :] - Sg_pred[:, :, :-1, :]
            Sg_grad_z = Sg_pred[:, :, :, 1:] - Sg_pred[:, :, :, :-1]
            max_grad = 0.3
            loss_darcy_x = torch.mean(F.relu(torch.abs(Sg_grad_x) - max_grad))
            loss_darcy_y = torch.mean(F.relu(torch.abs(Sg_grad_y) - max_grad))
            loss_darcy_z = torch.mean(F.relu(torch.abs(Sg_grad_z) - max_grad))
            mask_gas = (Sg_pred[:, :, :, :-1] > 0.01).float()
            loss_gravity = -torch.mean(Sg_grad_z * mask_gas)
            loss_gravity = torch.clamp(loss_gravity, min=0)
            loss_darcy = loss_darcy_x + loss_darcy_y + loss_darcy_z + loss_gravity * 0.1
        total_loss = (self.lambda_data * loss_data +
                     self.lambda_sat * (loss_sat + loss_bounds) +
                     self.lambda_mass * loss_mass +
                     self.lambda_darcy * loss_darcy)
        return total_loss, {
            'data': loss_data.item(),
            'sat': loss_sat.item(),
            'bounds': loss_bounds.item(),
            'mass': loss_mass.item() if isinstance(loss_mass, torch.Tensor) else 0.0,
            'darcy': loss_darcy.item() if isinstance(loss_darcy, torch.Tensor) else 0.0
        }

# ============================================================================
# TRAINING AVEC SCHEDULED SAMPLING
# ============================================================================

def train_epoch_scheduled(model, dataloader, optimizer, criterion, device, epoch, 
                          max_epochs, scheduled_sampling, use_rollout_training=False):
    model.train()
    total_loss = 0
    loss_components = {'data': 0, 'sat': 0, 'bounds': 0, 'mass': 0, 'darcy': 0}
    all_preds = []
    all_targets = []
    
    if use_rollout_training:
        sampling_prob = 0.0
    else:
        sampling_prob = scheduled_sampling.get_sampling_prob(epoch, max_epochs)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} (TF={sampling_prob:.3f})')
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape
        optimizer.zero_grad()
        spatial_sizes = [
            (nx, ny, nz),
            (nx//2, ny//2, nz//2),
            (nx//4, ny//4, nz//4),
            (nx//8, ny//8, nz//8)
        ]
        states = model.init_hidden_states(batch_size, spatial_sizes)
        total_batch_loss = 0
        predictions = []
        
        for t in range(seq_len):
            if t == 0:
                model_input = inputs[:, t]
            else:
                use_ground_truth = (torch.rand(1).item() < sampling_prob)
                if use_ground_truth:
                    model_input = inputs[:, t]
                else:
                    prev_pred = predictions[-1].detach()
                    Sg_max_prev = inputs[:, t, 3:4]
                    meta_prev = inputs[:, t, 4:5]
                    Sg_current = prev_pred[:, 0:1]
                    Sg_max_updated = torch.maximum(Sg_max_prev, Sg_current)
                    model_input = torch.cat([
                        prev_pred[:, 1:2],
                        prev_pred[:, 0:1],
                        prev_pred[:, 2:3],
                        Sg_max_updated,
                        meta_prev
                    ], dim=1)
            
            pred, states = model(model_input, states)
            predictions.append(pred)
            metadata = {
                'phi': batch['phi'].mean().item(),
                'K': batch['K'].mean().item(),
                'dt': 1.0,
                'operation': 1.0
            }
            loss, components = criterion(pred, targets[:, t], input_state=inputs[:, t], metadata=metadata)
            total_batch_loss += loss
            for key in loss_components:
                loss_components[key] += components[key]
            all_preds.append(pred[:, 0].detach().cpu().numpy().flatten())
            all_targets.append(targets[:, t, 0].cpu().numpy().flatten())
        
        total_batch_loss = total_batch_loss / seq_len
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += total_batch_loss.item()
        pbar.set_postfix({'loss': f'{total_batch_loss.item():.6f}'})
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(dataloader) * seq_len
    return total_loss / len(dataloader), {k: v/n for k, v in loss_components.items()}, r2, sampling_prob


def validate_convlstm(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape
            spatial_sizes = [
                (nx, ny, nz),
                (nx//2, ny//2, nz//2),
                (nx//4, ny//4, nz//4),
                (nx//8, ny//8, nz//8)
            ]
            states = model.init_hidden_states(batch_size, spatial_sizes)
            batch_loss = 0
            for t in range(seq_len):
                pred, states = model(inputs[:, t], states)
                loss, _ = criterion(pred, targets[:, t])
                batch_loss += loss
                all_preds.append(pred[:, 0].detach().cpu().numpy().flatten())
                all_targets.append(targets[:, t, 0].cpu().numpy().flatten())
            total_loss += (batch_loss / seq_len).item()
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return total_loss / len(dataloader), r2


def evaluate_rollout(model, dataloader, device, rollout_steps=10):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Rollout {rollout_steps}-step'):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape
            spatial_sizes = [
                (nx, ny, nz),
                (nx//2, ny//2, nz//2),
                (nx//4, ny//4, nz//4),
                (nx//8, ny//8, nz//8)
            ]
            states = model.init_hidden_states(batch_size, spatial_sizes)
            current_input = inputs[:, 0]
            for t in range(min(rollout_steps, seq_len)):
                pred, states = model(current_input, states)
                all_preds.append(pred[:, 0].cpu().numpy().flatten())
                all_targets.append(targets[:, t, 0].cpu().numpy().flatten())
                if t < seq_len - 1:
                    Sg_pred = pred[:, 0:1]
                    Sw_pred = pred[:, 1:2]
                    P_pred = pred[:, 2:3]
                    Sg_max_prev = inputs[:, t, 3:4]
                    Sg_max_new = torch.maximum(Sg_max_prev, Sg_pred)
                    meta = inputs[:, t, 4:5]
                    current_input = torch.cat([Sw_pred, Sg_pred, P_pred, Sg_max_new, meta], dim=1)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# ============================================================================
# MAIN - PROTECTION MULTIPROCESSING POUR MAC
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("ConvLSTM U-Net - SCHEDULED SAMPLING - MAC LOCAL")
    print("SPLIT 70/15/15 - SEED=42")
    print("Epochs 1-30: Scheduled Sampling (inverse_sigmoid k=5)")
    print("Epochs 31-40: Pure Rollout Training")
    print(f"{'='*70}\n")

    # Config
    fidelity = 'medium'
    sequence_length = 10
    batch_size = 1
    base_features = 32
    num_epochs = 40
    learning_rate = 1e-4
    num_workers = 0
    pin_memory = False

    print(f"Configuration:")
    print(f"  Fidelity: {fidelity}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Base features: {base_features}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Split: 70/15/15")
    print(f"  Seed: {SEED}")
    print(f"  Schedule: inverse_sigmoid (k=5)\n")

    # Dataset
    dataset = SequenceUHSDataset(DATA_PATH, fidelity, sequence_length)

    if len(dataset) == 0:
        raise ValueError("No sequences created!")

    # SPLIT 70/15/15 AVEC SEED
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"Split 70/15/15 (seed={SEED}):")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}\n")

    # Model
    model = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=base_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    # Loss & Optimizer
    criterion = PhysicsInformedLoss(lambda_data=1.0, lambda_sat=0.5, lambda_mass=0.2, lambda_darcy=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Scheduled Sampling
    scheduled_sampling = ScheduledSampling(schedule_type='inverse_sigmoid', k=5)

    # Training
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_r2': [], 'val_r2': [],
        'sampling_prob': []
    }

    # Reprise automatique
    checkpoint_file = CHECKPOINT_DIR / 'ablation_A_best.pth'
    start_epoch = 1

    if checkpoint_file.exists():
        print(f"\n{'='*70}")
        print("REPRISE DEPUIS CHECKPOINT")
        print(f"{'='*70}")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_r2 = checkpoint.get('best_val_r2', -float('inf'))
            
            history_file = CHECKPOINT_DIR / 'history_convlstm_SS_70_15_15_local.pkl'
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
            
            print(f"Reprise à l'epoch {start_epoch}")
            print(f"Meilleur val R²: {best_val_r2:.4f}\n")
        except Exception as e:
            print(f"Erreur chargement checkpoint: {e}")
            print("Redémarrage depuis epoch 1\n")
            start_epoch = 1

    # TRAINING LOOP
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*70}")
        
        if epoch <= 30:
            use_rollout = False
            print(f"Epoch {epoch}/{num_epochs} - SCHEDULED SAMPLING")
        else:
            use_rollout = True
            print(f"Epoch {epoch}/{num_epochs} - PURE ROLLOUT TRAINING")
        
        print(f"{'='*70}")

        train_loss, components, train_r2, sampling_prob = train_epoch_scheduled(
            model, train_loader, optimizer, criterion, device, epoch,
            30, scheduled_sampling, use_rollout_training=use_rollout
        )
        val_loss, val_r2 = validate_convlstm(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['sampling_prob'].append(sampling_prob)

        print(f"\nTrain: Loss={train_loss:.6f}, R²={train_r2:.4f}")
        print(f"  - Data: {components['data']:.6f}")
        print(f"  - Saturation: {components['sat']:.6f}")
        print(f"  - Mass: {components['mass']:.6f}")
        print(f"  - Darcy: {components['darcy']:.6f}")
        print(f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f}")
        print(f"Sampling prob (TF): {sampling_prob:.3f}")

        scheduler.step(val_loss)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'history': history
            }, CHECKPOINT_DIR / "best_convlstm_SS_70_15_15_local.pt")
            print(f"  ✓ Best model saved (R²={val_r2:.4f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_r2': val_r2,
            'best_val_loss': best_val_loss,
            'best_val_r2': best_val_r2,
            'history': history
        }, checkpoint_file)

        with open(CHECKPOINT_DIR / 'history_convlstm_SS_70_15_15_local.pkl', 'wb') as f:
            pickle.dump(history, f)

        if epoch % 5 == 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
            axes[0, 0].plot(history['val_loss'], 'r-', label='Val', linewidth=2)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            axes[0, 0].set_title('Loss - Scheduled Sampling')

            axes[0, 1].plot(history['train_r2'], 'b-', label='Train', linewidth=2)
            axes[0, 1].plot(history['val_r2'], 'r-', label='Val', linewidth=2)
            axes[0, 1].axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='Target')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            axes[0, 1].set_title(f'R² Score - Best: {best_val_r2:.4f}')

            axes[1, 0].plot(history['sampling_prob'], 'purple', linewidth=2)
            axes[1, 0].axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Switch to Rollout')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Teacher Forcing Probability')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].set_title('Scheduled Sampling Decay')
            axes[1, 0].set_ylim([0, 1])

            gap = [abs(t - v) for t, v in zip(history['train_r2'], history['val_r2'])]
            axes[1, 1].plot(gap, 'orange', linewidth=2)
            axes[1, 1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good gap')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('|Train R² - Val R²|')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].set_title('Generalization Gap')

            plt.tight_layout()
            plt.savefig(CHECKPOINT_DIR / f'convlstm_SS_70_15_15_local_epoch_{epoch}.png', dpi=150)
            plt.close()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'history': history
            }, CHECKPOINT_DIR / f'checkpoint_SS_70_15_15_local_epoch_{epoch}.pt')

            print(f"  💾 Checkpoint epoch {epoch} saved")

        gc.collect()

    # ÉVALUATION FINALE
    print(f"\n{'='*70}")
    print("ÉVALUATION FINALE SUR TEST SET")
    print(f"{'='*70}\n")

    best_ckpt = torch.load(CHECKPOINT_DIR / "best_convlstm_SS_70_15_15_local.pt", 
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_loss, test_r2_single = validate_convlstm(model, test_loader, criterion, device)
    print(f"Test Single-step (TF): Loss={test_loss:.6f}, R²={test_r2_single:.4f}")

    test_r2_rollout_10 = evaluate_rollout(model, test_loader, device, rollout_steps=10)
    print(f"Test Rollout 10-step:  R²={test_r2_rollout_10:.4f}")

    print(f"\n{'='*70}")
    print("RÉSUMÉ FINAL - SCHEDULED SAMPLING (LOCAL)")
    print(f"{'='*70}")
    print(f"  Best Epoch: {best_ckpt['epoch']}")
    print(f"  Best Val R²: {best_val_r2:.4f}")
    print(f"  Test Single-step R²: {test_r2_single:.4f}")
    print(f"  Test Rollout-10 R²: {test_r2_rollout_10:.4f}")
    print(f"{'='*70}\n")
