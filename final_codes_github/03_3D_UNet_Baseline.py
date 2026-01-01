"""
3D U-Net Baseline pour UHS
SANS mémoire temporelle (pas de LSTM)
Pour comparaison avec ConvLSTM U-Net

SPLIT 70/15/15 - SEED=42 (même que ConvLSTM pour comparaison équitable)
AVEC REPRISE AUTOMATIQUE DEPUIS CHECKPOINT
"""

# ============================================================================
# SETUP
# ============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

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
import json
from datetime import datetime

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

DATA_PATH = Path('/content/drive/MyDrive/Deeponet/multifidelity_deeponet_data')
CHECKPOINT_DIR = DATA_PATH / 'checkpoints' / 'baseline_3d_unet'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Data path exists: {DATA_PATH.exists()}")

# ============================================================================
# 3D U-NET BASELINE (SANS CONVLSTM)
# ============================================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_out = self.conv(x)
        pooled = self.pool(conv_out)
        return pooled, conv_out


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [diff_w//2, diff_w-diff_w//2,
                         diff_h//2, diff_h-diff_h//2,
                         diff_d//2, diff_d-diff_d//2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_features=32):
        super().__init__()
        self.enc1 = EncoderBlock3D(in_channels, base_features)
        self.enc2 = EncoderBlock3D(base_features, base_features * 2)
        self.enc3 = EncoderBlock3D(base_features * 2, base_features * 4)
        self.bottleneck = ConvBlock3D(base_features * 4, base_features * 8)
        self.dec3 = DecoderBlock3D(base_features * 8, base_features * 4, base_features * 4)
        self.dec2 = DecoderBlock3D(base_features * 4, base_features * 2, base_features * 2)
        self.dec1 = DecoderBlock3D(base_features * 2, base_features, base_features)
        self.output = nn.Conv3d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.base_features = base_features

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        bn = self.bottleneck(x3)
        d3 = self.dec3(bn, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        out = self.output(d1)
        out = self.sigmoid(out)
        return out


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
        except:
            return None

    def _create_sequences(self):
        sequences = []
        for mat_file in self.mat_files:
            result = self._load_mat_flexible(mat_file)
            if result is None:
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
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        result = self._load_mat_flexible(seq_info['file'])
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
    def __init__(self, lambda_data=1.0, lambda_sat=0.5, lambda_mass=0.2, lambda_darcy=0.1):
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
        loss_darcy = torch.tensor(0.0, device=pred.device)

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

            Sg_grad_x = Sg_pred[:, 1:, :, :] - Sg_pred[:, :-1, :, :]
            Sg_grad_y = Sg_pred[:, :, 1:, :] - Sg_pred[:, :, :-1, :]
            Sg_grad_z = Sg_pred[:, :, :, 1:] - Sg_pred[:, :, :, :-1]
            max_grad = 0.3
            loss_darcy = torch.mean(F.relu(torch.abs(Sg_grad_x) - max_grad))
            loss_darcy += torch.mean(F.relu(torch.abs(Sg_grad_y) - max_grad))
            loss_darcy += torch.mean(F.relu(torch.abs(Sg_grad_z) - max_grad))

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
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_unet(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {'data': 0, 'sat': 0, 'bounds': 0, 'mass': 0, 'darcy': 0}
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape

        optimizer.zero_grad()
        total_batch_loss = 0

        for t in range(seq_len):
            pred = model(inputs[:, t])
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
    return total_loss / len(dataloader), {k: v/n for k, v in loss_components.items()}, r2


def validate_unet(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape
            batch_loss = 0

            for t in range(seq_len):
                pred = model(inputs[:, t])
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


def evaluate_rollout_unet(model, dataloader, device, rollout_steps=10):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Rollout {rollout_steps}-step'):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            batch_size, seq_len, in_channels, nx, ny, nz = inputs.shape
            current_input = inputs[:, 0]

            for t in range(min(rollout_steps, seq_len)):
                pred = model(current_input)
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
# MAIN
# ============================================================================

print(f"\n{'='*70}")
print("3D U-NET BASELINE (SANS MÉMOIRE TEMPORELLE)")
print("AVEC REPRISE AUTOMATIQUE")
print("SPLIT 70/15/15 - SEED=42")
print(f"{'='*70}\n")

# Config
fidelity = 'medium'
sequence_length = 10
batch_size = 1
num_epochs = 40
learning_rate = 1e-4
base_features = 32

print(f"Configuration:")
print(f"  Model: 3D U-Net (baseline)")
print(f"  Epochs: {num_epochs}")
print(f"  Base features: {base_features}")
print(f"  Seed: {SEED}\n")

# Dataset
dataset = SequenceUHSDataset(DATA_PATH, fidelity, sequence_length)

# SPLIT 70/15/15
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Model
model = UNet3D(in_channels=5, out_channels=3, base_features=base_features).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Loss & Optimizer
criterion = PhysicsInformedLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ============================================================================
# REPRISE AUTOMATIQUE DEPUIS CHECKPOINT
# ============================================================================

checkpoint_file = CHECKPOINT_DIR / 'latest_3d_unet_baseline.pt'
start_epoch = 1
best_val_loss = float('inf')
best_val_r2 = -float('inf')
history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [], 'lr': []}

# DEBUG - Vérification des fichiers
print(f"\n🔍 DEBUG - Vérification des fichiers:")
print(f"Chemin checkpoint: {checkpoint_file}")
print(f"Dossier existe: {CHECKPOINT_DIR.exists()}")
print(f"Fichier existe: {checkpoint_file.exists()}")

import os
if CHECKPOINT_DIR.exists():
    print(f"\nFichiers dans le dossier:")
    for f in os.listdir(CHECKPOINT_DIR):
        fpath = CHECKPOINT_DIR / f
        size_mb = fpath.stat().st_size / (1024*1024) if fpath.exists() else 0
        print(f"  - {f} ({size_mb:.1f} MB)")

if checkpoint_file.exists():
    print(f"\n{'='*70}")
    print("🔄 REPRISE DEPUIS CHECKPOINT")
    print(f"{'='*70}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        # Afficher les clés disponibles
        print(f"Clés dans le checkpoint: {list(checkpoint.keys())}")

        # Charger le modèle
        model.load_state_dict(checkpoint['model_state_dict'])

        # Charger optimizer SI disponible
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer chargé")
        else:
            print("⚠️ Optimizer non trouvé, utilisation d'un nouvel optimizer")

        # Récupérer l'epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        best_val_r2 = checkpoint.get('best_val_r2', checkpoint.get('val_r2', -float('inf')))

        # Charger historique
        if 'history' in checkpoint:
            history = checkpoint['history']
        else:
            # Essayer le fichier local d'abord, puis le fichier standard
            history_file = CHECKPOINT_DIR / 'history_3d_unet_baseline_local.pkl'
            if not history_file.exists():
                history_file = CHECKPOINT_DIR / 'history_3d_unet_baseline.pkl'
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)

        print(f"✓ Reprise à l'epoch {start_epoch}")
        print(f"✓ Meilleur val R²: {best_val_r2:.4f}")
        print(f"✓ Historique: {len(history.get('train_loss', []))} epochs chargés\n")

    except Exception as e:
        print(f"⚠️ Erreur chargement: {e}")
        print("Redémarrage depuis epoch 1\n")
        start_epoch = 1
else:
    print("📝 Aucun checkpoint trouvé, démarrage depuis epoch 1\n")
# ============================================================================
# TRAINING LOOP (avec reprise)
# ============================================================================

for epoch in range(start_epoch, num_epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{num_epochs} - 3D U-Net Baseline")
    print(f"{'='*70}")

    train_loss, components, train_r2 = train_epoch_unet(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, val_r2 = validate_unet(model, val_loader, criterion, device)

    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_r2'].append(train_r2)
    history['val_r2'].append(val_r2)
    history['lr'].append(current_lr)

    print(f"\nTrain: Loss={train_loss:.6f}, R²={train_r2:.4f}")
    print(f"  - Data: {components['data']:.6f}")
    print(f"  - Saturation: {components['sat']:.6f}")
    print(f"  - Mass: {components['mass']:.6f}")
    print(f"  - Darcy: {components['darcy']:.6f}")
    print(f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f}")
    print(f"LR: {current_lr:.2e}")

    scheduler.step(val_loss)

    # Sauvegarder meilleur modèle
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
        }, CHECKPOINT_DIR / "best_3d_unet_baseline.pt")
        print(f"  ✓ Best model saved (R²={val_r2:.4f})")

    # Checkpoint latest (TOUJOURS sauvegarder pour reprise)
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

    # Sauvegarder historique
    with open(CHECKPOINT_DIR / 'history_3d_unet_baseline.pkl', 'wb') as f:
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
        axes[0, 0].set_title('Loss - 3D U-Net Baseline')

        axes[0, 1].plot(history['train_r2'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(history['val_r2'], 'r-', label='Val', linewidth=2)
        axes[0, 1].axhline(y=0.99, color='green', linestyle='--', alpha=0.5)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].set_title(f'R² Score - Best: {best_val_r2:.4f}')

        axes[1, 0].plot(history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_title('Learning Rate')

        gap = [abs(t - v) for t, v in zip(history['train_r2'], history['val_r2'])]
        axes[1, 1].plot(gap, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Train R² - Val R²|')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_title('Generalization Gap')

        plt.tight_layout()
        plt.savefig(CHECKPOINT_DIR / f'3d_unet_baseline_epoch_{epoch}.png', dpi=150)
        plt.show()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_r2': val_r2,
            'history': history
        }, CHECKPOINT_DIR / f'checkpoint_3d_unet_baseline_epoch_{epoch}.pt')

        print(f"  💾 Checkpoint epoch {epoch} saved")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# ÉVALUATION FINALE
# ============================================================================

print(f"\n{'='*70}")
print("ÉVALUATION FINALE SUR TEST SET")
print(f"{'='*70}\n")

best_ckpt = torch.load(CHECKPOINT_DIR / "best_3d_unet_baseline.pt", weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])

test_loss, test_r2_single = validate_unet(model, test_loader, criterion, device)
print(f"Test Single-step: R²={test_r2_single:.4f}")

test_r2_rollout_10 = evaluate_rollout_unet(model, test_loader, device, rollout_steps=10)
print(f"Test Rollout 10-step: R²={test_r2_rollout_10:.4f}")

final_results = {
    'model': '3D_UNet_baseline',
    'seed': SEED,
    'split': '70/15/15',
    'best_epoch': best_ckpt['epoch'],
    'best_val_r2': float(best_val_r2),
    'test_single_step_r2': float(test_r2_single),
    'test_rollout_10_r2': float(test_r2_rollout_10),
    'n_params': n_params,
}

with open(CHECKPOINT_DIR / 'final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n{'='*70}")
print("RÉSUMÉ FINAL - 3D U-NET BASELINE")
print(f"{'='*70}")
print(f"  Best Epoch: {best_ckpt['epoch']}")
print(f"  Best Val R²: {best_val_r2:.4f}")
print(f"  Test Single-step R²: {test_r2_single:.4f}")
print(f"  Test Rollout-10 R²: {test_r2_rollout_10:.4f}")
print(f"{'='*70}\n")
