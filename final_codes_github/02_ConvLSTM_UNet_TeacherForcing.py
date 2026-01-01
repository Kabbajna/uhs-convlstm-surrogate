"""
ConvLSTM U-Net complet pour Google Colab
Architecture avec mémoire temporelle pour UHS
SPLIT 70/15/15 avec SEED=42 (reproductible)
"""

# ============================================================================
# SETUP + SEED POUR REPRODUCTIBILITÉ
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
CHECKPOINT_DIR = DATA_PATH / 'checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Data path exists: {DATA_PATH.exists()}")

# ============================================================================
# ConvLSTM CELL
# ============================================================================

class ConvLSTMCell3D(nn.Module):
    """ConvLSTM cell pour données 3D avec mémoire temporelle"""

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
# LOSS PHYSICS-INFORMED COMPLÈTE
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_sat=0.5, lambda_mass=0.2, lambda_darcy=0.1):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_sat = lambda_sat
        self.lambda_mass = lambda_mass
        self.lambda_darcy = lambda_darcy

    def forward(self, pred, target, input_state=None, metadata=None):
        """
        Args:
            pred: [batch, 3, nx, ny, nz] - prédictions (Sg, Sw, P)
            target: [batch, 3, nx, ny, nz] - vérité terrain
            input_state: [batch, 5, nx, ny, nz] - état au temps t (pour calculer dérivées temporelles)
            metadata: dict avec 'phi', 'K', 'dt', 'operation'
        """
        # 1. Data loss
        loss_data = F.mse_loss(pred, target)

        # 2. Saturation constraints: Sw + Sg = 1
        Sg_pred = pred[:, 0]
        Sw_pred = pred[:, 1]
        loss_sat = torch.mean((Sg_pred + Sw_pred - 1.0) ** 2)

        # Physical bounds: 0 <= Sg, Sw <= 1
        loss_bounds = torch.mean(F.relu(-Sg_pred)) + torch.mean(F.relu(Sg_pred - 1.0))
        loss_bounds += torch.mean(F.relu(-Sw_pred)) + torch.mean(F.relu(Sw_pred - 1.0))

        # 3. Conservation de masse (équation de continuité)
        loss_mass = torch.tensor(0.0, device=pred.device)

        if input_state is not None and metadata is not None:
            phi = metadata.get('phi', 0.2)
            dt = metadata.get('dt', 1.0)

            # État au temps t
            Sg_t = input_state[:, 1]  # Channel 1 = Sg

            # ∂(φ·Sg)/∂t ≈ (φ·Sg_{t+1} - φ·Sg_t) / Δt
            dSg_dt = (Sg_pred - Sg_t) / (dt + 1e-8)

            # Conservation: ∂(φ·Sg)/∂t + ∇·(ρ_g·v_g) = Q
            # Simplifié: le changement de masse doit être cohérent avec l'opération
            operation = metadata.get('operation', 0.5)  # 1=injection, 0=production

            if operation > 0.5:  # Injection
                # Pendant injection, Sg doit augmenter (dSg_dt > 0)
                loss_mass_dir = torch.mean(F.relu(-dSg_dt))  # Pénalise si dSg_dt < 0
            else:  # Production
                # Pendant production, Sg doit diminuer (dSg_dt < 0)
                loss_mass_dir = torch.mean(F.relu(dSg_dt))  # Pénalise si dSg_dt > 0

            # Le taux de changement doit être physiquement plausible
            # |∂(φ·Sg)/∂t| ne doit pas être trop grand
            loss_mass_rate = torch.mean(torch.abs(phi * dSg_dt) ** 2) * 0.01

            loss_mass = loss_mass_dir + loss_mass_rate

        # 4. Loi de Darcy (contraintes sur les gradients spatiaux)
        loss_darcy = torch.tensor(0.0, device=pred.device)

        if input_state is not None and metadata is not None:
            K = metadata.get('K', 1e-13)

            # Calculer gradients spatiaux de la pression
            P_pred = pred[:, 2]

            # Gradients en x, y, z
            grad_x = P_pred[:, 1:, :, :] - P_pred[:, :-1, :, :]
            grad_y = P_pred[:, :, 1:, :] - P_pred[:, :, :-1, :]
            grad_z = P_pred[:, :, :, 1:] - P_pred[:, :, :, :-1]

            # Loi de Darcy: v = -(K/μ)·∇P
            # Les gradients de pression doivent être cohérents avec le déplacement de Sg

            # Gradient de Sg en x, y, z
            Sg_grad_x = Sg_pred[:, 1:, :, :] - Sg_pred[:, :-1, :, :]
            Sg_grad_y = Sg_pred[:, :, 1:, :] - Sg_pred[:, :, :-1, :]
            Sg_grad_z = Sg_pred[:, :, :, 1:] - Sg_pred[:, :, :, :-1]

            # Le gaz migre selon le gradient de pression ET la gravité (vers le haut)
            # Contrainte: ∇Sg doit être aligné avec -∇P (le gaz va vers basse pression)
            # et avec +z (le gaz monte par flottabilité)

            # Limiter les changements spatiaux brusques (stabilité numérique)
            max_grad = 0.3  # Changement maximal de saturation entre cellules voisines
            loss_darcy_x = torch.mean(F.relu(torch.abs(Sg_grad_x) - max_grad))
            loss_darcy_y = torch.mean(F.relu(torch.abs(Sg_grad_y) - max_grad))
            loss_darcy_z = torch.mean(F.relu(torch.abs(Sg_grad_z) - max_grad))

            # Contrainte gravitaire: en absence de pression, Sg doit diminuer vers le bas
            # (le gaz monte, donc Sg augmente avec z)
            # Favoriser ∂Sg/∂z > 0 dans les zones avec Sg > 0
            mask_gas = (Sg_pred[:, :, :, :-1] > 0.01).float()
            loss_gravity = -torch.mean(Sg_grad_z * mask_gas)  # Favorise gradient positif en z
            loss_gravity = torch.clamp(loss_gravity, min=0)  # Seulement si négatif

            loss_darcy = loss_darcy_x + loss_darcy_y + loss_darcy_z + loss_gravity * 0.1

        # Total loss
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
# TRAINING
# ============================================================================

def train_epoch_convlstm(model, dataloader, optimizer, criterion, device, epoch):
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

        spatial_sizes = [
            (nx, ny, nz),
            (nx//2, ny//2, nz//2),
            (nx//4, ny//4, nz//4),
            (nx//8, ny//8, nz//8)
        ]
        states = model.init_hidden_states(batch_size, spatial_sizes)

        total_batch_loss = 0

        for t in range(seq_len):
            pred, states = model(inputs[:, t], states)

            # Métadonnées enrichies pour la loss physics
            metadata = {
                'phi': batch['phi'].mean().item(),
                'K': batch['K'].mean().item(),
                'dt': 1.0,  # Normalised timestep
                'operation': 1.0  # Assume injection par défaut, à adapter si disponible
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


# ============================================================================
# MAIN
# ============================================================================

print(f"\n{'='*70}")
print("ConvLSTM U-Net Training - MEDIUM Fidelity")
print("SPLIT 70/15/15 - SEED=42 (reproductible)")
print(f"{'='*70}\n")

# Config
fidelity = 'medium'
sequence_length = 10
batch_size = 1
num_epochs = 40
learning_rate = 1e-4

print(f"Configuration:")
print(f"  Fidelity: {fidelity}")
print(f"  Sequence length: {sequence_length}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {num_epochs}")
print(f"  Base features: 32")
print(f"  Split: 70/15/15")
print(f"  Seed: {SEED}\n")

# Dataset
dataset = SequenceUHSDataset(DATA_PATH, fidelity, sequence_length)

if len(dataset) == 0:
    raise ValueError("No sequences created!")

# ============================================================================
# SPLIT 70/15/15 AVEC SEED
# ============================================================================
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f"Split 70/15/15 (seed={SEED}):")
print(f"  Train: {len(train_dataset)}")
print(f"  Val:   {len(val_dataset)}")
print(f"  Test:  {len(test_dataset)}\n")

# Model
model = ConvLSTMUNet3D(in_channels=5, out_channels=3, base_features=32).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}\n")

# Loss & Optimizer
criterion = PhysicsInformedLoss(lambda_data=1.0, lambda_sat=0.5, lambda_mass=0.2, lambda_darcy=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training
best_val_loss = float('inf')
best_val_r2 = -float('inf')
history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}

# ============================================================================
# PAS DE REPRISE - FRESH START
# ============================================================================

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"{'='*70}")

    train_loss, components, train_r2 = train_epoch_convlstm(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, val_r2 = validate_convlstm(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_r2'].append(train_r2)
    history['val_r2'].append(val_r2)

    print(f"\nTrain: Loss={train_loss:.6f}, R²={train_r2:.4f}")
    print(f"  - Data: {components['data']:.6f}")
    print(f"  - Saturation: {components['sat']:.6f}")
    print(f"  - Mass: {components['mass']:.6f}")
    print(f"  - Darcy: {components['darcy']:.6f}")
    print(f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f}")

    scheduler.step(val_loss)

    # Sauvegarder meilleur modèle (basé sur R² maintenant)
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
        }, CHECKPOINT_DIR / "best_convlstm_70_15_15.pt")
        print(f"  ✓ Best model saved (R²={val_r2:.4f})")

    # Sauvegarder checkpoint latest (pour reprise)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_r2': val_r2,
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2,
        'history': history
    }, CHECKPOINT_DIR / "latest_convlstm_70_15_15.pt")

    # Sauvegarder historique
    with open(CHECKPOINT_DIR / 'history_convlstm_70_15_15.pkl', 'wb') as f:
        pickle.dump(history, f)

    if epoch % 5 == 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_title('Loss - ConvLSTM U-Net (70/15/15)')

        ax2.plot(history['train_r2'], 'b-', label='Train', linewidth=2)
        ax2.plot(history['val_r2'], 'r-', label='Val', linewidth=2)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target')
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_title(f'R² Score - Best: {best_val_r2:.4f}')

        plt.tight_layout()
        plt.savefig(CHECKPOINT_DIR / f'convlstm_70_15_15_epoch_{epoch}.png', dpi=150)
        plt.show()

        # Sauvegarder checkpoint numéroté
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_r2': val_r2,
            'history': history
        }, CHECKPOINT_DIR / f'checkpoint_convlstm_70_15_15_epoch_{epoch}.pt')

        print(f"  💾 Checkpoint epoch {epoch} saved")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# ÉVALUATION FINALE SUR TEST SET
# ============================================================================

print(f"\n{'='*70}")
print("ÉVALUATION FINALE SUR TEST SET")
print(f"{'='*70}\n")

# Charger le meilleur modèle
best_ckpt = torch.load(CHECKPOINT_DIR / "best_convlstm_70_15_15.pt", weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])

test_loss, test_r2 = validate_convlstm(model, test_loader, criterion, device)
print(f"Test: Loss={test_loss:.6f}, R²={test_r2:.4f}")

print(f"\n{'='*70}")
print("RÉSUMÉ FINAL")
print(f"{'='*70}")
print(f"  Best Epoch: {best_ckpt['epoch']}")
print(f"  Best Val R²: {best_val_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"{'='*70}\n")
