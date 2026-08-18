"""
Trainer pour l'entraînement de l'encodeur-décodeur RAVEN.

Gère la boucle d'entraînement, la validation, l'évaluation,
et le suivi de l'historique des pertes.
"""

import torch
import torch.nn.functional as F


class Trainer:
    """
    Gestionnaire d'entraînement pour le modèle RAVEN.

    Paramètres
    ----------
    model : nn.Module
        Le modèle à entraîner (RavenEncoderDecoder).
    optimizer : torch.optim.Optimizer
        L'optimiseur (ex: Adam).
    criterion : nn.Module
        La fonction de perte (CompositeLoss).
    device : torch.device
        Le périphérique (CPU ou GPU).
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, images, targets):
        """
        Exécute une étape d'entraînement sur un batch.

        Paramètres
        ----------
        images : Tensor (B, 1, 160, 160)
        targets : Tensor (B, 10, 13)

        Retourne
        --------
        dict avec les pertes : 'total', 'presence', 'attr', 'recon' (floats)
        """
        self.model.train()
        images = images.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        pred_z, pred_img = self.model(images)
        losses = self.criterion(pred_z, pred_img, targets, images)
        total_loss = losses['total']

        total_loss.backward()
        self.optimizer.step()

        return {
            "total": losses['total'].item(),
            "presence": losses['presence'].item(),
            "attr": losses['attr'].item(),
            "recon": losses['recon'].item()
        }

    def evaluation_step(self, images, targets):
        """
        Exécute une étape d'évaluation sur un batch (sans gradients).

        Paramètres
        ----------
        images : Tensor (B, 1, 160, 160)
        targets : Tensor (B, 10, 13)

        Retourne
        --------
        dict avec les pertes : 'total', 'presence', 'attr', 'recon' (floats)
        """
        self.model.eval()
        images = images.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            pred_z, pred_img = self.model(images)
            losses = self.criterion(pred_z, pred_img, targets, images)

        return {
            "total": losses['total'].item(),
            "presence": losses['presence'].item(),
            "attr": losses['attr'].item(),
            "recon": losses['recon'].item()
        }

    def fit(self, train_loader, val_loader, epochs, checkpoint_dir=None):
        """
        Boucle d'entraînement complète sur plusieurs époques.

        Paramètres
        ----------
        train_loader : DataLoader
            Chargeur de données d'entraînement.
        val_loader : DataLoader
            Chargeur de données de validation.
        epochs : int
            Nombre d'époques.
        checkpoint_dir : str, optional
            Répertoire pour sauvegarder les checkpoints.

        Retourne
        --------
        dict
            Historique des pertes avec les clés :
            'train_total', 'train_presence', 'train_attr', 'train_recon',
            'val_total', 'val_presence', 'val_attr', 'val_recon'
            Chaque valeur est une liste de floats (une par époque).
        """
        history = {
            'train_total': [], 'train_presence': [], 'train_attr': [], 'train_recon': [],
            'val_total': [], 'val_presence': [], 'val_attr': [], 'val_recon': []
        }

        for epoch in range(epochs):
            # --- Entraînement ---
            train_sums = {'total': 0.0, 'presence': 0.0, 'attr': 0.0, 'recon': 0.0}
            num_train_batches = 0

            for batch in train_loader:
                images, targets = batch[0], batch[1]
                losses = self.train_step(images, targets)
                for key in train_sums:
                    train_sums[key] += losses[key]
                num_train_batches += 1

            # Moyennes d'entraînement
            for key in train_sums:
                avg = train_sums[key] / max(num_train_batches, 1)
                history[f'train_{key}'].append(avg)

            # --- Validation ---
            val_sums = {'total': 0.0, 'presence': 0.0, 'attr': 0.0, 'recon': 0.0}
            num_val_batches = 0

            for batch in val_loader:
                images, targets = batch[0], batch[1]
                losses = self.evaluation_step(images, targets)
                for key in val_sums:
                    val_sums[key] += losses[key]
                num_val_batches += 1

            # Moyennes de validation
            for key in val_sums:
                avg = val_sums[key] / max(num_val_batches, 1)
                history[f'val_{key}'].append(avg)

            # Affichage
            t = history['train_total'][-1]
            p = history['train_presence'][-1]
            a = history['train_attr'][-1]
            r = history['train_recon'][-1]
            v = history['val_total'][-1]
            print(
                f"Epoque {epoch+1}/{epochs} - "
                f"Train: {t:.4f} (p={p:.4f} a={a:.4f} r={r:.4f}) - "
                f"Val: {v:.4f}"
            )

            # Checkpoint toutes les 5 époques
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  > Checkpoint sauvegarde : {path}")

        return history

    def evaluate(self, test_loader):
        """
        Évalue le modèle sur un jeu de test et calcule des métriques détaillées.

        Paramètres
        ----------
        test_loader : DataLoader
            Chargeur de données de test.

        Retourne
        --------
        dict
            Métriques : 'loss_total', 'loss_presence', 'loss_attr', 'loss_recon',
            'presence_accuracy', 'type_accuracy', 'size_mae', 'color_mae', 'angle_mae',
            'recon_mse'
        """
        self.model.eval()

        # Accumulateurs pour les pertes
        loss_sums = {'total': 0.0, 'presence': 0.0, 'attr': 0.0, 'recon': 0.0}
        num_batches = 0

        # Accumulateurs pour les métriques
        presence_correct = 0
        presence_total = 0
        type_correct = 0
        type_total = 0
        size_ae_sum = 0.0
        color_ae_sum = 0.0
        angle_ae_sum = 0.0
        regression_count = 0
        recon_mse_sum = 0.0
        recon_count = 0

        with torch.no_grad():
            for batch in test_loader:
                images, targets = batch[0].to(self.device), batch[1].to(self.device)
                pred_z, pred_img = self.model(images)

                # Pertes
                losses = self.criterion(pred_z, pred_img, targets, images)
                for key in loss_sums:
                    loss_sums[key] += losses[key].item()
                num_batches += 1

                # --- Métriques de présence ---
                pred_presence = (torch.sigmoid(pred_z[:, :, 0]) > 0.5).float()
                target_presence = targets[:, :, 0]
                presence_correct += (pred_presence == target_presence).sum().item()
                presence_total += target_presence.numel()

                # --- Métriques sur les slots actifs ---
                mask = (target_presence == 1)
                if mask.any():
                    # Précision du type (argmax sur colonnes 2:8)
                    pred_types = pred_z[mask][:, 2:8].argmax(dim=1)
                    target_types = targets[mask][:, 2:8].argmax(dim=1)
                    type_correct += (pred_types == target_types).sum().item()
                    type_total += target_types.numel()

                    # MAE des attributs de régression
                    pred_attrs = torch.sigmoid(pred_z[mask])
                    size_ae_sum += (pred_attrs[:, 8] - targets[mask][:, 8]).abs().sum().item()
                    color_ae_sum += (pred_attrs[:, 9] - targets[mask][:, 9]).abs().sum().item()
                    angle_ae_sum += (pred_attrs[:, 12] - targets[mask][:, 12]).abs().sum().item()
                    regression_count += mask.sum().item()

                # Reconstruction MSE
                recon_mse_sum += F.mse_loss(pred_img, images, reduction='sum').item()
                recon_count += images.numel()

        metrics = {
            'loss_total': loss_sums['total'] / max(num_batches, 1),
            'loss_presence': loss_sums['presence'] / max(num_batches, 1),
            'loss_attr': loss_sums['attr'] / max(num_batches, 1),
            'loss_recon': loss_sums['recon'] / max(num_batches, 1),
            'presence_accuracy': presence_correct / max(presence_total, 1),
            'type_accuracy': type_correct / max(type_total, 1),
            'size_mae': size_ae_sum / max(regression_count, 1),
            'color_mae': color_ae_sum / max(regression_count, 1),
            'angle_mae': angle_ae_sum / max(regression_count, 1),
            'recon_mse': recon_mse_sum / max(recon_count, 1),
        }

        return metrics