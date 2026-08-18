"""
Fixtures partagées pour les tests du projet HRI-GNN.
"""

import pytest
import torch
from pathlib import Path

from data.raven_dataset import RavenDataset


# =============================================================================
# FIXTURES GLOBALES
# =============================================================================

@pytest.fixture
def device():
    """Sélection automatique du périphérique (GPU si disponible, sinon CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dataset_path():
    """Chemin vers les données brutes RAVEN."""
    path = Path(__file__).resolve().parent.parent / "data" / "raw"
    assert path.exists(), f"Dataset introuvable : {path}"
    return str(path)


@pytest.fixture
def dataset_single(dataset_path):
    """Dataset avec 1 seul fichier NPZ, configuration center_single."""
    return RavenDataset(dataset_path, split="train", limit_per_config=1, config="center_single")
