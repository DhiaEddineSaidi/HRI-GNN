"""
Tests pour le module models.encoder_decoder (RavenEncoderDecoder).

Couvre :
    - Architecture du modèle (création, paramètres)
    - Encodeur (shapes de sortie, types, tailles de batch variables)
    - Forward pass complet (encodage + reconstruction)
    - Étape d'entraînement avec Trainer + CompositeLoss
    - Intégration avec le dataset RAVEN réel
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.encoder_decoder import RavenEncoderDecoder
from losses.composite_loss import CompositeLoss
from training.trainer import Trainer


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def model(device):
    """Modèle RavenEncoderDecoder sur le périphérique disponible."""
    return RavenEncoderDecoder(num_slots=10, feat_dim=13).to(device)


@pytest.fixture
def optimizer(model):
    """Optimiseur Adam pour le modèle."""
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def criterion():
    """Fonction de perte composite."""
    return CompositeLoss()


@pytest.fixture
def trainer(model, optimizer, criterion, device):
    """Instance de Trainer configurée pour les tests."""
    return Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)


@pytest.fixture
def dummy_batch(device):
    """Batch synthétique pour les tests unitaires (pas besoin du dataset)."""
    batch_size = 4
    images = torch.randn(batch_size, 1, 160, 160).to(device)
    targets = torch.zeros(batch_size, 10, 13).to(device)
    # Simuler 1 objet présent par sample dans le slot 0
    targets[:, 0, 0] = 1.0         # Présence
    targets[:, 0, 2] = 1.0         # Type one-hot (triangle) — colonne 2
    targets[:, 0, 8] = 0.5         # Size
    targets[:, 0, 9] = 0.3         # Color
    targets[:, 0, 10] = 0.4        # Position X
    targets[:, 0, 11] = 0.6        # Position Y
    targets[:, 0, 12] = 0.25       # Angle
    return images, targets


# =============================================================================
# TESTS DE L'ARCHITECTURE
# =============================================================================

class TestArchitectureModele:
    """Tests pour la structure du modèle."""

    def test_creation_modele(self, model):
        """Le modèle se crée sans erreur."""
        assert model is not None
        assert model.num_slots == 10
        assert model.feat_dim == 13

    def test_nombre_parametres(self, model):
        """Le modèle doit avoir des paramètres entraînables."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0
        assert trainable_params == total_params  # Tout doit être entraînable

    def test_dimensions_personnalisees(self, device):
        """On peut créer un modèle avec des dimensions personnalisées."""
        model = RavenEncoderDecoder(num_slots=5, feat_dim=8).to(device)
        assert model.num_slots == 5
        assert model.feat_dim == 8


# =============================================================================
# TESTS DE L'ENCODEUR
# =============================================================================

class TestEncodeur:
    """Tests pour la partie encodeur (backbone + projection)."""

    def test_shape_sortie_encodeur(self, model, dummy_batch):
        """L'encodeur doit produire [B, 10, 13]."""
        images, _ = dummy_batch
        model.eval()
        with torch.no_grad():
            z = model.encode(images)
        assert z.shape == (images.size(0), 10, 13)

    def test_type_sortie_encodeur(self, model, dummy_batch):
        """La sortie de l'encodeur doit être float32."""
        images, _ = dummy_batch
        model.eval()
        with torch.no_grad():
            z = model.encode(images)
        assert z.dtype == torch.float32

    def test_tailles_batch_variables(self, model, device):
        """L'encodeur doit gérer différentes tailles de batch."""
        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 1, 160, 160).to(device)
            model.eval()
            with torch.no_grad():
                z = model.encode(x)
            assert z.shape == (batch_size, 10, 13)

    def test_image_unique(self, model, device):
        """L'encodeur fonctionne avec une seule image."""
        x = torch.randn(1, 1, 160, 160).to(device)
        model.eval()
        with torch.no_grad():
            z = model.encode(x)
        assert z.shape == (1, 10, 13)


# =============================================================================
# TESTS DU FORWARD PASS COMPLET
# =============================================================================

class TestForwardPass:
    """Tests pour le forward pass complet (encodage + décodage)."""

    def test_shapes_sorties(self, model, dummy_batch):
        """forward() retourne (z, reconstruction) avec les bonnes shapes."""
        images, _ = dummy_batch
        model.eval()
        with torch.no_grad():
            z, recon = model(images)
        assert z.shape == (images.size(0), 10, 13), f"Shape z : {z.shape}"
        assert recon.shape == images.shape, f"Shape reconstruction : {recon.shape}"

    def test_plage_reconstruction(self, model, dummy_batch):
        """La reconstruction doit être dans [0, 1] grâce au Sigmoid final."""
        images, _ = dummy_batch
        model.eval()
        with torch.no_grad():
            _, recon = model(images)
        assert recon.min() >= 0.0, f"Min reconstruction : {recon.min()}"
        assert recon.max() <= 1.0, f"Max reconstruction : {recon.max()}"

    def test_flux_gradients(self, model, dummy_batch):
        """Les gradients doivent circuler du décodeur à l'encodeur."""
        images, _ = dummy_batch
        z, recon = model(images)
        loss = recon.sum()
        loss.backward()
        # Vérifier que le backbone reçoit des gradients
        first_conv = model.backbone[0]
        assert first_conv.weight.grad is not None, "Le backbone ne reçoit pas de gradients"
        assert first_conv.weight.grad.abs().sum() > 0


# =============================================================================
# TESTS DE L'ÉTAPE D'ENTRAÎNEMENT (via Trainer)
# =============================================================================

class TestEtapeEntrainement:
    """Tests pour l'étape d'entraînement avec Trainer + CompositeLoss."""

    def test_train_step_retourne_pertes(self, trainer, dummy_batch):
        """train_step doit retourner un dictionnaire de pertes."""
        images, targets = dummy_batch
        pertes = trainer.train_step(images, targets)
        assert isinstance(pertes, dict)
        assert "total" in pertes
        assert "presence" in pertes
        assert "attr" in pertes
        assert "recon" in pertes
        assert isinstance(pertes["total"], float)
        assert pertes["total"] > 0

    def test_perte_diminue(self, trainer, dummy_batch):
        """Après quelques steps, la perte doit diminuer."""
        images, targets = dummy_batch
        pertes_liste = []
        for _ in range(10):
            pertes = trainer.train_step(images, targets)
            pertes_liste.append(pertes["total"])
        # La perte devrait globalement diminuer
        assert pertes_liste[-1] < pertes_liste[0], (
            f"La perte n'a pas diminué : {pertes_liste[0]:.4f} → {pertes_liste[-1]:.4f}"
        )

    def test_mise_a_jour_parametres(self, model, trainer, dummy_batch):
        """Les paramètres du modèle doivent changer après un step."""
        images, targets = dummy_batch
        # Sauvegarder les poids avant
        poids_avant = {nom: p.clone() for nom, p in model.named_parameters()}
        trainer.train_step(images, targets)
        # Vérifier qu'au moins un paramètre a changé
        modifie = False
        for nom, p in model.named_parameters():
            if not torch.equal(poids_avant[nom], p):
                modifie = True
                break
        assert modifie, "Aucun paramètre n'a été mis à jour"

    def test_train_step_sans_objets(self, trainer, device):
        """train_step fonctionne même si aucun objet n'est présent (target tout à 0)."""
        images = torch.randn(2, 1, 160, 160).to(device)
        targets = torch.zeros(2, 10, 13).to(device)
        pertes = trainer.train_step(images, targets)
        assert isinstance(pertes["total"], float)
        assert not torch.isnan(torch.tensor(pertes["total"])), "La perte est NaN"


# =============================================================================
# TESTS D'INTÉGRATION AVEC LE DATASET
# =============================================================================

class TestIntegrationDataset:
    """Tests d'intégration modèle + dataset réel."""

    def test_modele_accepte_sortie_dataset(self, model, dataset_single, device):
        """Le modèle accepte directement la sortie du dataset."""
        image, target, p_type = dataset_single[0]
        image = image.unsqueeze(0).to(device)  # [1, 1, 160, 160]
        model.eval()
        with torch.no_grad():
            z, recon = model(image)
        assert z.shape == (1, 10, 13)

    def test_batch_dataloader_dans_modele(self, model, dataset_single, device):
        """Un batch du DataLoader passe dans le modèle sans erreur."""
        loader = DataLoader(dataset_single, batch_size=4, shuffle=False)
        images, targets, p_types = next(iter(loader))
        images = images.to(device)
        model.eval()
        with torch.no_grad():
            z, recon = model(images)
        assert z.shape == (4, 10, 13)

    def test_train_step_donnees_reelles(self, trainer, dataset_single, device):
        """Un step d'entraînement fonctionne avec les données réelles."""
        loader = DataLoader(dataset_single, batch_size=4, shuffle=False)
        images, targets, p_types = next(iter(loader))
        images = images.to(device)
        targets = targets.to(device)
        pertes = trainer.train_step(images, targets)
        assert isinstance(pertes["total"], float)
        assert pertes["total"] > 0
