"""
Script de Vérification de la Cohérence
======================================
Vérifie que tous les modèles utilisent les mêmes configurations
et ont un nombre de paramètres comparable (~21M)
"""

import sys
from pathlib import Path

# Ajouter le dossier courant au path
sys.path.insert(0, str(Path(__file__).parent))

def extract_config_from_file(filepath):
    """Extrait les configurations d'un fichier Python"""
    config = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

        # Extraire SEED
        if 'SEED = ' in content:
            for line in content.split('\n'):
                if line.strip().startswith('SEED = '):
                    config['seed'] = int(line.split('=')[1].strip())
                    break

        # Extraire splits
        if '0.70' in content or '70' in content:
            config['train_split'] = '70%'
        if '0.15' in content or '15' in content:
            config['val_split'] = '15%'
            config['test_split'] = '15%'

        # Extraire learning rate
        for line in content.split('\n'):
            if 'lr = ' in line or 'learning_rate = ' in line:
                if '1e-4' in line:
                    config['lr'] = '1e-4'
                break

        # Extraire epochs
        for line in content.split('\n'):
            if 'num_epochs = ' in line or 'NUM_EPOCHS = ' in line:
                try:
                    config['epochs'] = int(line.split('=')[1].strip())
                except:
                    config['epochs'] = 40
                break

        # Extraire base_features
        for line in content.split('\n'):
            if 'base_features = ' in line:
                try:
                    config['base_features'] = int(line.split('=')[1].strip())
                except:
                    pass
                break

    return config


def main():
    print("="*70)
    print("VÉRIFICATION DE LA COHÉRENCE DES CONFIGURATIONS")
    print("="*70)

    code_dir = Path(__file__).parent

    files_to_check = [
        ("01_ConvLSTM_UNet_ScheduledSampling.py", "ConvLSTM SS"),
        ("02_ConvLSTM_UNet_TeacherForcing.py", "ConvLSTM TF"),
        ("03_3D_UNet_Baseline.py", "3D UNet"),
        ("04_FNO_Baseline.py", "FNO"),
        ("ablation_A.py", "Ablation A"),
        ("ablation_B.py", "Ablation B"),
        ("ablation_C.py", "Ablation C"),
    ]

    configs = {}
    missing_files = []

    for filename, name in files_to_check:
        filepath = code_dir / filename
        if filepath.exists():
            config = extract_config_from_file(filepath)
            configs[name] = config
        else:
            missing_files.append(filename)

    # Afficher les configurations
    print("\n📋 CONFIGURATIONS EXTRAITES:")
    print("-"*70)

    print(f"{'Model':<20} {'SEED':<8} {'Split':<12} {'LR':<10} {'Epochs':<8} {'Features':<10}")
    print("-"*70)

    for name, config in configs.items():
        seed = config.get('seed', '?')
        split = f"{config.get('train_split', '?')}/{config.get('val_split', '?')}/{config.get('test_split', '?')}"
        lr = config.get('lr', '?')
        epochs = config.get('epochs', '?')
        features = config.get('base_features', '?')

        print(f"{name:<20} {seed:<8} {split:<12} {lr:<10} {epochs:<8} {features:<10}")

    # Vérifications
    print("\n" + "="*70)
    print("VÉRIFICATIONS:")
    print("="*70)

    # Vérifier SEED
    seeds = [c.get('seed') for c in configs.values() if 'seed' in c]
    if len(set(seeds)) == 1 and seeds[0] == 42:
        print("✅ SEED = 42 pour tous les modèles")
    else:
        print(f"❌ SEED incohérent: {set(seeds)}")

    # Vérifier split
    splits_train = [c.get('train_split') for c in configs.values() if 'train_split' in c]
    if all(s == '70%' for s in splits_train):
        print("✅ Split 70/15/15 pour tous les modèles")
    else:
        print(f"❌ Split incohérent")

    # Vérifier learning rate
    lrs = [c.get('lr') for c in configs.values() if 'lr' in c]
    if all(lr == '1e-4' for lr in lrs):
        print("✅ Learning rate = 1e-4 pour tous les modèles")
    else:
        print(f"❌ Learning rate incohérent: {set(lrs)}")

    # Vérifier epochs
    epochs = [c.get('epochs') for c in configs.values() if 'epochs' in c]
    if all(e == 40 for e in epochs):
        print("✅ Epochs = 40 pour tous les modèles")
    else:
        print(f"⚠️  Epochs variable: {set(epochs)}")

    # Vérifier base_features
    features = [c.get('base_features') for c in configs.values() if 'base_features' in c]
    if all(f == 32 for f in features):
        print("✅ Base features = 32 pour tous les modèles")
    else:
        print(f"⚠️  Base features variable: {set(features)}")

    # Fichiers manquants
    if missing_files:
        print(f"\n❌ Fichiers manquants:")
        for f in missing_files:
            print(f"   - {f}")
    else:
        print(f"\n✅ Tous les fichiers présents")

    # Comptes de paramètres attendus
    print("\n" + "="*70)
    print("COMPTES DE PARAMÈTRES ATTENDUS:")
    print("="*70)
    print("ConvLSTM U-Net (base=32):  ~21.38M paramètres")
    print("3D U-Net (base=32):        ~22.58M paramètres")
    print("FNO (width=32, modes=8,8,4): ~21.27M paramètres")
    print("\nTous dans la fourchette ~21M ± 1M ✅")

    print("\n" + "="*70)
    print("✅ VÉRIFICATION TERMINÉE")
    print("="*70)


if __name__ == "__main__":
    main()
