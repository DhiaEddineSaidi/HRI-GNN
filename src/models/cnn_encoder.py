from torch.nn.modules import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.raven_dataset import RavenDataset

class RavenEncoderDecoder(nn.Module):
    def __init__(self, num_slots=10, feat_dim=13):
        super(RavenEncoderDecoder, self).__init__()
        self.num_slots = num_slots   # 1 Slot "Out" + 9 Slots "In"
        self.feat_dim = feat_dim     # [p, type(6), size, color, x, y, sin, cos] = 13
        
        # --- 1. ENCODEUR (BACKBONE CNN) ---
        # Extrait les caractéristiques (features) spatiales de l'image 160x160
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 80x80
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 40x40
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 20x20
            nn.AdaptiveAvgPool2d((5, 5)), # Sortie fixe 128x5x5
            nn.Flatten() # Sortie fixe 3200
        )
        
        # --- 2. COUCHE DE PROJECTION VERS L'ESPACE LATENT ---
        # Couche de projection vers l'espace latent (Vecteur de propriétés), c'est ici qu'on transforme les 6400 features (256*5*5) en 130 features (10*13)
        self.projection_layer = nn.Sequential(
            nn.Linear(128*5*5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_slots * feat_dim) 
        )
        
        # --- 3. DÉCODEUR (RECONSTRUCTION de l'espace latent vers l'image) ---
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
        features = self.backbone(x)
        latent = self.projection_layer(features)
        return latent.view(-1, self.num_slots, self.feat_dim)

    def forward(self, x):
        z = self.encode(x)
        z_flat = z.reshape(z.size(0), -1)
        d = self.decoder_input(z_flat).view(-1, 128, 5, 5)
        recon = self.decoder_cnn(d)
        return z, recon

# --- 4. FONCTION DE PERTE HYBRIDE (SUPERVISÉE + RECONSTRUCTION) ---

def train_step(model, images, targets, optimizer):
    """
    images: [B, 1, 160, 160]
    targets: [B, 10, 13] -> [Presence(1), Type(6), Size(1), Color(1), Pos(2), SinCos(2)]
    """
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward pass
    pred_z, pred_img = model(images)
    
    # 2. Perte de PRÉSENCE (BCE) - Indispensable pour le "Number"
    # pred_z[:,:,0] sont les logits de présence
    loss_presence = F.binary_cross_entropy_with_logits(pred_z[:, :, 0], targets[:, :, 0])
    
    # 3. Perte d'ATTRIBUTS (Supervision NPZ)
    # On crée un masque : on n'entraîne que sur les objets qui existent (Presence=1)
    mask = (targets[:, :, 0] == 1)
    loss_attr = 0
    
    if mask.any():
        # Classification du Type (6 classes)
        target_types = targets[mask][:, 1:7].argmax(dim=1)
        loss_attr += F.cross_entropy(pred_z[mask][:, 1:7], target_types)
        
        # Régression (Size, Color, Pos, Angle) - Valeurs entre 0 et 1
        # On applique sigmoid sur les prédictions pour matcher l'échelle
        pred_attrs = torch.sigmoid(pred_z[mask][:, 7:])
        loss_attr += F.mse_loss(pred_attrs, targets[mask][:, 7:])

    # 4. Perte de RECONSTRUCTION (L'assurance contre le bruit)
    loss_recon = F.mse_loss(pred_img, images)
    
    # 5. Calcul total avec pondération
    # On booste la reconstruction (x20) pour forcer la précision spatiale
    total_loss = loss_presence + loss_attr + (20.0 * loss_recon)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

