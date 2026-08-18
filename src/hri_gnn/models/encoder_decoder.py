"""
Encodeur-Décodeur CNN pour le pipeline Neuro-Symbolique RAVEN.

Ce module implémente un auto-encodeur convolutionnel qui :
  1. Encode une image RAVEN 160×160 en un vecteur de propriétés structuré (10 slots × 13 attributs)
  2. Reconstruit l'image à partir de ce vecteur pour valider la qualité de l'encodage

Architecture :
    Image [1, 160, 160]  →  Backbone CNN  →  Projection  →  z [10, 13]
                                                              ↓
                                                          Décodeur  →  Image reconstruite [1, 160, 160]

Chaque slot du vecteur z représente une entité visuelle avec :
    [présence(1), number(1), type_one_hot(6), size(1), color(1), pos_x(1), pos_y(1), angle(1)] = 13
"""

import torch.nn as nn


class RavenEncoderDecoder(nn.Module):
    """
    Auto-encodeur CNN pour extraire les propriétés structurées des panels RAVEN.

    Paramètres
    ----------
    num_slots : int
        Nombre de slots d'entités (1 slot "Out" + 9 slots "In" = 10 par défaut).
    feat_dim : int
        Dimension du vecteur d'attributs par slot (13 par défaut).
    """

    def __init__(self, num_slots=10, feat_dim=13):
        super().__init__()
        self.num_slots = num_slots
        self.feat_dim = feat_dim

        # --- 1. ENCODEUR (Backbone CNN) ---
        # Extrait les caractéristiques spatiales de l'image 160×160
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # → 80×80
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # → 40×40
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),# → 20×20
            nn.AdaptiveAvgPool2d((5, 5)),  # → 128×5×5
            nn.Flatten()                    # → 3200
        )

        # --- 2. PROJECTION vers l'espace latent ---
        # Transforme les 3200 features en (num_slots × feat_dim) = 130 attributs structurés
        self.projection_layer = nn.Sequential(
            nn.Linear(128 * 5 * 5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_slots * feat_dim)
        )

        # --- 3. DÉCODEUR (Reconstruction latent → image) ---
        # Reconstruit l'image à partir du vecteur de propriétés pour valider l'encodage
        self.decoder_input = nn.Linear(num_slots * feat_dim, 128 * 5 * 5)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),  # 5→10
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),   # 10→20
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # 20→40
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),   # 40→80
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(), # 80→160
        )

    def encode(self, x):
        """
        Encode une image en vecteur de propriétés structuré.

        Paramètres
        ----------
        x : Tensor de shape (B, 1, 160, 160)

        Retourne
        --------
        Tensor de shape (B, num_slots, feat_dim)
        """
        features = self.backbone(x)
        latent = self.projection_layer(features)
        return latent.view(-1, self.num_slots, self.feat_dim)

    def forward(self, x):
        """
        Passe avant complète : encodage + reconstruction.

        Paramètres
        ----------
        x : Tensor de shape (B, 1, 160, 160)

        Retourne
        --------
        z : Tensor de shape (B, num_slots, feat_dim) — vecteur latent structuré
        recon : Tensor de shape (B, 1, 160, 160) — image reconstruite
        """
        z = self.encode(x)
        z_flat = z.reshape(z.size(0), -1)
        d = self.decoder_input(z_flat).view(-1, 128, 5, 5)
        recon = self.decoder_cnn(d)
        return z, recon
