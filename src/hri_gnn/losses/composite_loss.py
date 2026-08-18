"""
Fonction de perte composite pour l'entraînement de l'encodeur-décodeur RAVEN.

Combine trois composantes :
    1. Perte de PRÉSENCE (BCE) — détection des objets présents
    2. Perte d'ATTRIBUTS — classification du type + régression des propriétés
    3. Perte de RECONSTRUCTION — fidélité de l'image reconstruite

Format du vecteur target (10 slots × 13 attributs) :
    Col 0     : Présence (0 ou 1)
    Col 1     : Number (normalisé 0-1)
    Col 2-7   : Type one-hot (6 classes : none, tri, squ, pen, hex, cir)
    Col 8     : Size (normalisé 0-1)
    Col 9     : Color (normalisé 0-1)
    Col 10-11 : Position X, Y
    Col 12    : Angle (normalisé 0-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Perte composite multi-objectifs pour l'auto-encodeur RAVEN.

    Paramètres
    ----------
    recon_weight : float
        Poids de la perte de reconstruction (défaut: 20.0).
        Valeur élevée pour forcer la précision spatiale de l'encodage.
    """

    def __init__(self, recon_weight=20.0):
        super(CompositeLoss, self).__init__()
        self.recon_weight = recon_weight

    def forward(self, pred_z, pred_img, targets, images):
        """
        Calcule la perte composite.

        Paramètres
        ----------
        pred_z : Tensor (B, 10, 13)
            Prédictions du vecteur latent structuré.
        pred_img : Tensor (B, 1, 160, 160)
            Image reconstruite par le décodeur.
        targets : Tensor (B, 10, 13)
            Vecteurs cibles extraits du XML.
        images : Tensor (B, 1, 160, 160)
            Images originales.

        Retourne
        --------
        dict avec les clés : 'total', 'presence', 'attr', 'recon'
        """
        device = pred_z.device

        # 1. Perte de PRÉSENCE (BCE) — indispensable pour le "Number"
        # pred_z[:, :, 0] sont les logits de présence
        loss_presence = F.binary_cross_entropy_with_logits(
            pred_z[:, :, 0], targets[:, :, 0]
        )

        # 2. Perte d'ATTRIBUTS (supervision structurelle)
        # Masque : on n'entraîne que sur les objets existants (Présence=1)
        mask = (targets[:, :, 0] == 1)
        loss_attr = torch.tensor(0.0, device=device)

        if mask.any():
            # Classification du Type (6 classes, colonnes 2:8)
            target_types = targets[mask][:, 2:8].argmax(dim=1)
            loss_attr = loss_attr + F.cross_entropy(pred_z[mask][:, 2:8], target_types)

            # Régression des attributs continus : Number(1), Size(8), Color(9), PosX(10), PosY(11), Angle(12)
            # On applique sigmoid sur les prédictions pour les ramener dans [0, 1]
            regression_cols = [1, 8, 9, 10, 11, 12]
            pred_attrs = torch.sigmoid(pred_z[mask][:, regression_cols])
            target_attrs = targets[mask][:, regression_cols]
            loss_attr = loss_attr + F.mse_loss(pred_attrs, target_attrs)

        # 3. Perte de RECONSTRUCTION (assurance contre le bruit d'encodage)
        loss_recon = F.mse_loss(pred_img, images)

        # 4. Calcul total avec pondération
        total_loss = loss_presence + loss_attr + (self.recon_weight * loss_recon)

        return {
            "total": total_loss,
            "presence": loss_presence,
            "attr": loss_attr,
            "recon": loss_recon
        }