"""
Tests pour le module data.raven_dataset (RavenDataset).

Couvre :
    - Initialisation et chargement du dataset
    - Accès aux éléments (__getitem__)
    - Parsing du vecteur cible depuis le XML
    - Compatibilité avec le DataLoader PyTorch
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from data.raven_dataset import RavenDataset


# =============================================================================
# FIXTURES (dataset_path et dataset_single sont dans conftest.py)
# =============================================================================

@pytest.fixture
def dataset_multi(dataset_path):
    """Dataset avec 3 fichiers NPZ, configuration center_single."""
    return RavenDataset(dataset_path, split="train", limit_per_config=3, config="center_single")


# =============================================================================
# TESTS D'INITIALISATION
# =============================================================================

class TestDatasetInit:
    """Tests pour la création et l'initialisation du dataset."""

    def test_dataset_loads(self, dataset_single):
        """Le dataset se charge sans erreur."""
        assert dataset_single is not None

    def test_dataset_len_single_file(self, dataset_single):
        """1 fichier NPZ = 16 panneaux."""
        assert len(dataset_single) == 16

    def test_dataset_len_multi_files(self, dataset_multi):
        """3 fichiers NPZ = 48 panneaux."""
        assert len(dataset_multi) == 3 * 16

    def test_dataset_samples_structure(self, dataset_single):
        """Chaque sample est un tuple (archive_path, panel_idx)."""
        for archive_path, panel_idx in dataset_single.samples:
            assert isinstance(archive_path, str)
            assert archive_path.endswith(".npz")
            assert 0 <= panel_idx < 16

    def test_invalid_split_raises(self, dataset_path):
        """Un split inexistant ne doit pas trouver de fichiers."""
        with pytest.raises(ValueError, match="Aucun fichier trouvé"):
            RavenDataset(dataset_path, split="nonexistent", config="center_single")

    def test_invalid_data_dir_raises(self):
        """Un répertoire inexistant doit lever une erreur."""
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            RavenDataset("/chemin/qui/nexiste/pas", split="train")


# =============================================================================
# TESTS DE __getitem__
# =============================================================================

class TestDatasetGetItem:
    """Tests pour l'accès aux éléments du dataset."""

    def test_getitem_returns_three_values(self, dataset_single):
        """__getitem__ retourne (image, target_vector, panel_type)."""
        result = dataset_single[0]
        assert len(result) == 3

    def test_image_shape(self, dataset_single):
        """L'image doit être un tenseur [1, 160, 160]."""
        image, _, _ = dataset_single[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 160, 160)

    def test_image_dtype_float32(self, dataset_single):
        """L'image doit être en float32 (normalisée)."""
        image, _, _ = dataset_single[0]
        assert image.dtype == torch.float32

    def test_image_normalized(self, dataset_single):
        """Les valeurs de pixels doivent être dans [0, 1]."""
        image, _, _ = dataset_single[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_target_vector_shape(self, dataset_single):
        """Le vecteur cible doit être [10, 13] (10 slots x 13 attributs)."""
        _, target, _ = dataset_single[0]
        assert isinstance(target, torch.Tensor)
        assert target.shape == (10, 13)

    def test_target_vector_dtype_float32(self, dataset_single):
        """Le vecteur cible doit être en float32."""
        _, target, _ = dataset_single[0]
        assert target.dtype == torch.float32

    def test_panel_type_context(self, dataset_single):
        """Les panneaux 0-7 sont de type 'context'."""
        for idx in range(8):
            _, _, p_type = dataset_single[idx]
            assert p_type == "context", f"Panel {idx} devrait être 'context', got '{p_type}'"

    def test_panel_type_candidates_and_solution(self, dataset_single):
        """Les panneaux 8-15 sont 'candidate' ou 'solution', avec exactement 1 solution."""
        types = []
        for idx in range(8, 16):
            _, _, p_type = dataset_single[idx]
            assert p_type in ("candidate", "solution"), f"Panel {idx} type invalide: '{p_type}'"
            types.append(p_type)
        assert types.count("solution") == 1, "Il doit y avoir exactement 1 solution"
        assert types.count("candidate") == 7, "Il doit y avoir exactement 7 candidats"

    def test_index_out_of_range_raises(self, dataset_single):
        """Un index hors limites doit lever une erreur."""
        with pytest.raises(IndexError):
            dataset_single[16]

    def test_negative_index(self, dataset_single):
        """Les index négatifs doivent fonctionner (dernier élément)."""
        image, target, p_type = dataset_single[-1]
        assert image.shape == (1, 160, 160)
        assert target.shape == (10, 13)


# =============================================================================
# TESTS DE _parse_target
# =============================================================================

class TestParseTarget:
    """Tests pour le parsing du vecteur cible depuis le XML."""

    def test_presence_bit_set(self, dataset_single):
        """Au moins un slot doit avoir le bit de présence à 1."""
        _, target, _ = dataset_single[0]
        presence = target[:, 0]
        assert presence.sum() > 0, "Au moins un objet doit être présent"

    def test_type_one_hot(self, dataset_single):
        """Pour les slots présents, le type doit être one-hot (exactement un 1)."""
        _, target, _ = dataset_single[0]
        for slot_idx in range(10):
            if target[slot_idx, 0] == 1.0:  # slot présent
                type_vector = target[slot_idx, 2:8]
                assert type_vector.sum() == 1.0, (
                    f"Slot {slot_idx}: le type doit être one-hot, got {type_vector}"
                )

    def test_values_normalized(self, dataset_single):
        """Les attributs normalisés doivent être dans [0, 1]."""
        _, target, _ = dataset_single[0]
        for slot_idx in range(10):
            if target[slot_idx, 0] == 1.0:
                # Size (col 8), Color (col 9), Angle (col 12)
                assert 0.0 <= target[slot_idx, 8] <= 1.0, f"Size hors bornes au slot {slot_idx}"
                assert 0.0 <= target[slot_idx, 9] <= 1.0, f"Color hors bornes au slot {slot_idx}"
                assert 0.0 <= target[slot_idx, 12] <= 1.0, f"Angle hors bornes au slot {slot_idx}"

    def test_absent_slots_are_zeros(self, dataset_single):
        """Les slots sans objet doivent être tout à 0."""
        _, target, _ = dataset_single[0]
        for slot_idx in range(10):
            if target[slot_idx, 0] == 0.0:
                assert target[slot_idx].sum() == 0.0, (
                    f"Slot {slot_idx} absent mais contient des valeurs non-nulles"
                )

    def test_target_consistency_across_panels(self, dataset_single):
        """Chaque panneau doit produire un target valide et non-identique aux autres."""
        targets = []
        for idx in range(16):
            _, target, _ = dataset_single[idx]
            assert target.shape == (10, 13)
            targets.append(target)
        # Vérifier qu'au moins certains targets diffèrent (les panneaux ne sont pas tous identiques)
        all_same = all(torch.equal(targets[0], t) for t in targets[1:])
        assert not all_same, "Tous les targets ne devraient pas être identiques"


# =============================================================================
# TESTS DATALOADER COMPATIBILITÉ
# =============================================================================

class TestDataLoader:
    """Tests d'intégration avec PyTorch DataLoader."""

    def test_dataloader_batching(self, dataset_single):
        """Le dataset doit être compatible avec DataLoader."""
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset_single, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        images, targets, p_types = batch
        assert images.shape == (4, 1, 160, 160)
        assert targets.shape == (4, 10, 13)
        assert len(p_types) == 4

    def test_dataloader_full_epoch(self, dataset_single):
        """On peut itérer sur tout le dataset sans erreur."""
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset_single, batch_size=8, shuffle=False)
        total_samples = 0
        for images, targets, p_types in loader:
            total_samples += images.size(0)
        assert total_samples == 16
