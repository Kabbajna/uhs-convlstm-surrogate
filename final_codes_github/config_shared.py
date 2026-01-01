"""
CONFIGURATION PARTAGÉE POUR TOUS LES MODÈLES
===================================================
GARANTIT LA COHÉRENCE ENTRE TOUS LES MODÈLES POUR COMPARAISON ÉQUITABLE

Utilisé par:
- ConvLSTM U-Net avec Scheduled Sampling
- ConvLSTM U-Net avec Teacher Forcing only
- 3D U-Net Baseline (sans mémoire temporelle)
- FNO 3D Baseline

Version: LOCAL MAC M1
"""

import random
import numpy as np
import torch
from pathlib import Path

# ============================================================================
# SEED POUR REPRODUCTIBILITÉ EXACTE
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Fixe tous les seeds pour reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 SEED = {seed} (résultats reproductibles)")

# ============================================================================
# CHEMINS LOCAL MAC
# ============================================================================
DATA_PATH = Path('/Users/narjisse/Documents/Effat Courses/deeponet')
CHECKPOINT_DIR = DATA_PATH / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SPLIT DES DONNÉES (IDENTIQUE POUR TOUS)
# ============================================================================
TRAIN_SPLIT = 0.70  # 70% pour training
VAL_SPLIT = 0.15    # 15% pour validation
TEST_SPLIT = 0.15   # 15% pour test

# ============================================================================
# HYPERPARAMÈTRES PARTAGÉS
# ============================================================================

# Architecture (pour avoir ~21M paramètres dans chaque modèle)
BASE_FEATURES = 32  # Pour ConvLSTM U-Net et 3D U-Net
FNO_WIDTH = 32      # Pour FNO
FNO_MODES = [8, 8, 4]  # modes1, modes2, modes3

# Training
NUM_EPOCHS = 40
BATCH_SIZE = 1  # Nécessaire pour 3D avec mémoire limitée
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Scheduled Sampling (pour ConvLSTM SS uniquement)
SCHEDULED_SAMPLING_K = 5
PURE_ROLLOUT_START_EPOCH = 31  # Epochs 31-40 en pure autoregressive

# Séquences
SEQUENCE_LENGTH_TRAIN = 10  # Pendant training
SEQUENCE_LENGTH_TEST = 33   # Pendant évaluation (15 mois complets)

# Physics-informed loss weights
LAMBDA_SAT = 0.5   # Saturation conservation
LAMBDA_MASS = 0.2  # Mass conservation
LAMBDA_DARCY = 0.1 # Darcy flow

# Gradient clipping
GRAD_CLIP_NORM = 1.0

# ============================================================================
# DEVICE CONFIGURATION (MAC M1)
# ============================================================================

def get_device():
    """Détecte et retourne le device approprié pour Mac M1"""
    if torch.backends.mps.is_available():
        # MPS ne supporte pas ConvTranspose3d, donc on utilise CPU
        device = torch.device("cpu")
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
    return device

# ============================================================================
# COMPTE DE PARAMÈTRES
# ============================================================================

def count_parameters(model):
    """Compte le nombre de paramètres entraînables"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Nombre de paramètres: {total:,}")
    print(f"   ({total/1e6:.2f}M paramètres)")
    return total

# ============================================================================
# VÉRIFICATION DE LA COHÉRENCE
# ============================================================================

def verify_config():
    """Vérifie que la configuration est cohérente"""
    print("\n" + "="*70)
    print("CONFIGURATION PARTAGÉE - VÉRIFICATION")
    print("="*70)
    print(f"SEED: {SEED}")
    print(f"Split: {TRAIN_SPLIT:.0%} / {VAL_SPLIT:.0%} / {TEST_SPLIT:.0%}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length (train): {SEQUENCE_LENGTH_TRAIN}")
    print(f"Sequence length (test): {SEQUENCE_LENGTH_TEST}")
    print(f"\nArchitecture:")
    print(f"  Base features (ConvLSTM/UNet): {BASE_FEATURES}")
    print(f"  FNO width: {FNO_WIDTH}")
    print(f"  FNO modes: {FNO_MODES}")
    print(f"\nLoss weights:")
    print(f"  λ_sat: {LAMBDA_SAT}")
    print(f"  λ_mass: {LAMBDA_MASS}")
    print(f"  λ_darcy: {LAMBDA_DARCY}")
    print(f"\nData path: {DATA_PATH}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print("="*70 + "\n")

    # Vérifier que les splits totalisent 100%
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, "Splits doivent totaliser 100%"

    return True

if __name__ == "__main__":
    verify_config()
    set_seed()
    device = get_device()
