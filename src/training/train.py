"""
Script principal d'entraînement de l'encodeur-décodeur RAVEN.

Entraîne le modèle RavenEncoderDecoder sur toutes les configurations RAVEN,
trace les courbes d'apprentissage et évalue les performances sur le jeu de test.

Usage :
    python -m training.train
"""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour la sauvegarde d'images
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.encoder_decoder import RavenEncoderDecoder
from data.raven_dataset import RavenDataset
from losses.composite_loss import CompositeLoss
from training.trainer import Trainer


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LIMIT_PER_CONFIG = 200  # Fichiers NPZ par configuration (None = tout)


# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def plot_learning_curves(history, save_path):
    """
    Trace les courbes d'apprentissage (pertes) et les sauvegarde.

    Paramètres
    ----------
    history : dict
        Historique des pertes retourné par Trainer.fit().
    save_path : str
        Chemin pour sauvegarder l'image.
    """
    epochs = range(1, len(history['train_total']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a2e')
    fig.suptitle("Courbes d'apprentissage — RavenEncoderDecoder",
                 fontsize=16, fontweight='bold', color='#e0e0e0', y=0.98)

    titres = [
        ('total', 'Perte Totale'),
        ('presence', 'Perte Présence (BCE)'),
        ('attr', 'Perte Attributs (CE + MSE)'),
        ('recon', 'Perte Reconstruction (MSE)')
    ]

    for ax, (key, titre) in zip(axes.flat, titres):
        ax.set_facecolor('#2a2a4a')
        ax.plot(epochs, history[f'train_{key}'], '-o', color='#64ffda',
                linewidth=2, markersize=4, label='Entraînement')
        ax.plot(epochs, history[f'val_{key}'], '-s', color='#ff6b6b',
                linewidth=2, markersize=4, label='Validation')
        ax.set_title(titre, fontsize=12, color='#e0e0e0', fontweight='bold')
        ax.set_xlabel('Époque', color='#b0b0b0')
        ax.set_ylabel('Perte', color='#b0b0b0')
        ax.legend(facecolor='#2a2a4a', edgecolor='#555', labelcolor='#e0e0e0')
        ax.grid(True, alpha=0.2, color='#555')
        ax.tick_params(colors='#b0b0b0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  [OK] Courbes sauvegardees : {save_path}")


def print_metrics(metrics):
    """Affiche les métriques d'évaluation de façon formatée."""
    print("\n" + "=" * 60)
    print("  RESULTATS D'EVALUATION SUR LE JEU DE TEST")
    print("=" * 60)
    print(f"  Perte totale         : {metrics['loss_total']:.4f}")
    print(f"  Perte presence       : {metrics['loss_presence']:.4f}")
    print(f"  Perte attributs      : {metrics['loss_attr']:.4f}")
    print(f"  Perte reconstruction : {metrics['loss_recon']:.6f}")
    print("-" * 60)
    print(f"  Precision presence   : {metrics['presence_accuracy']*100:.2f}%")
    print(f"  Precision type       : {metrics['type_accuracy']*100:.2f}%")
    print(f"  MAE taille           : {metrics['size_mae']:.4f}")
    print(f"  MAE couleur          : {metrics['color_mae']:.4f}")
    print(f"  MAE angle            : {metrics['angle_mae']:.4f}")
    print(f"  MSE reconstruction   : {metrics['recon_mse']:.6f}")
    print("=" * 60)


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def run_training():
    """Configure et lance l'entraînement complet."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0. Périphérique
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Peripherique : {device}")
    if device.type == 'cuda':
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # 1. Données (toutes les configurations)
    print(f"\nChargement des donnees depuis : {DATA_PATH}")
    print(f"  Limite par config : {LIMIT_PER_CONFIG or 'Tout'}")

    train_dataset = RavenDataset(data_dir=DATA_PATH, split='train', limit_per_config=LIMIT_PER_CONFIG)
    val_dataset = RavenDataset(data_dir=DATA_PATH, split='val', limit_per_config=LIMIT_PER_CONFIG)
    test_dataset = RavenDataset(data_dir=DATA_PATH, split='test', limit_per_config=LIMIT_PER_CONFIG)

    print(f"  Train : {len(train_dataset)} panels ({len(train_dataset)//16} fichiers)")
    print(f"  Val   : {len(val_dataset)} panels ({len(val_dataset)//16} fichiers)")
    print(f"  Test  : {len(test_dataset)} panels ({len(test_dataset)//16} fichiers)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Modèle
    model = RavenEncoderDecoder(num_slots=10, feat_dim=13).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModele : RavenEncoderDecoder ({total_params:,} parametres)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CompositeLoss(recon_weight=20.0)

    # 3. Trainer
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    # 4. Entraînement
    print(f"\n{'='*60}")
    print(f"  ENTRAINEMENT - {EPOCHS} epoques, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"{'='*60}\n")

    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    history = trainer.fit(train_loader, val_loader, epochs=EPOCHS, checkpoint_dir=checkpoint_dir)

    # 5. Sauvegarder l'historique
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  [OK] Historique sauvegarde : {history_path}")

    # 6. Courbes d'apprentissage
    curves_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plot_learning_curves(history, curves_path)

    # 7. Sauvegarde finale du modèle
    model_path = os.path.join(OUTPUT_DIR, "raven_encoder_decoder_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  [OK] Modele sauvegarde : {model_path}")

    # Sauvegarde de l'encodeur seul (pour le GNN)
    encoder_path = os.path.join(OUTPUT_DIR, "raven_encoder_final.pth")
    encoder_state = {
        'backbone': model.backbone.state_dict(),
        'projection': model.projection_layer.state_dict()
    }
    torch.save(encoder_state, encoder_path)
    print(f"  [OK] Encodeur sauvegarde : {encoder_path}")

    # 8. Évaluation sur le jeu de test
    print("\nEvaluation sur le jeu de test...")
    metrics = trainer.evaluate(test_loader)
    print_metrics(metrics)

    # Sauvegarder les métriques
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  [OK] Metriques sauvegardees : {metrics_path}")

    return history, metrics


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    run_training()