"""
Tests pour le module utils.utils.

Couvre :
    - raven_xml_file_parser : parsing des fichiers XML RAVEN
    - build_panel_text_info : construction du texte descriptif des panneaux
"""

import pytest
import glob
import os
from pathlib import Path

from utils.utils import raven_xml_file_parser, build_panel_text_info


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def xml_path_center_single():
    """Chemin vers un fichier XML de la configuration center_single."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    pattern = str(data_dir / "center_single" / "RAVEN_*_train.xml")
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"Aucun fichier XML trouvé : {pattern}"
    return files[0]


@pytest.fixture
def xml_path_in_out():
    """Chemin vers un fichier XML de la configuration in_center_single_out_center_single (2 composants)."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    pattern = str(data_dir / "in_center_single_out_center_single" / "RAVEN_*_train.xml")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip("Configuration in_center_single_out_center_single non disponible")
    return files[0]


@pytest.fixture
def xml_data_center(xml_path_center_single):
    """Données XML parsées pour center_single."""
    return raven_xml_file_parser(xml_path_center_single)


@pytest.fixture
def xml_data_in_out(xml_path_in_out):
    """Données XML parsées pour in_center_single_out_center_single."""
    return raven_xml_file_parser(xml_path_in_out)


# =============================================================================
# TESTS DE raven_xml_file_parser
# =============================================================================

class TestRavenXmlFileParser:
    """Tests pour la fonction de parsing XML."""

    def test_retourne_dict(self, xml_data_center):
        """Le parser doit retourner un dictionnaire."""
        assert isinstance(xml_data_center, dict)

    def test_cles_principales(self, xml_data_center):
        """Le dictionnaire doit contenir 'panels' et 'rules'."""
        assert 'panels' in xml_data_center
        assert 'rules' in xml_data_center

    def test_nombre_panels(self, xml_data_center):
        """Un fichier RAVEN contient exactement 16 panneaux."""
        assert len(xml_data_center['panels']) == 16

    def test_structure_panel(self, xml_data_center):
        """Chaque panneau doit contenir une clé 'struct'."""
        for i, panel in enumerate(xml_data_center['panels']):
            assert 'struct' in panel, f"Panneau {i} : clé 'struct' manquante"

    def test_structure_struct(self, xml_data_center):
        """Chaque struct doit contenir 'name' et 'components'."""
        for i, panel in enumerate(xml_data_center['panels']):
            struct = panel['struct']
            assert 'name' in struct or 'components' in struct, (
                f"Panneau {i} : struct doit avoir 'name' ou 'components'"
            )
            assert 'components' in struct, f"Panneau {i} : 'components' manquant"
            assert isinstance(struct['components'], list)

    def test_composants_non_vides(self, xml_data_center):
        """Chaque panneau doit avoir au moins 1 composant."""
        for i, panel in enumerate(xml_data_center['panels']):
            components = panel['struct']['components']
            assert len(components) >= 1, f"Panneau {i} : aucun composant"

    def test_structure_composant(self, xml_data_center):
        """Chaque composant doit avoir 'name' et 'layout'."""
        panel = xml_data_center['panels'][0]
        for comp in panel['struct']['components']:
            assert 'name' in comp, "Composant sans 'name'"
            assert 'layout' in comp, "Composant sans 'layout'"

    def test_structure_layout(self, xml_data_center):
        """Le layout doit contenir 'Number', 'entities', et d'autres attributs."""
        panel = xml_data_center['panels'][0]
        layout = panel['struct']['components'][0]['layout']
        assert 'Number' in layout, "Layout sans 'Number'"
        assert 'entities' in layout, "Layout sans 'entities'"
        assert isinstance(layout['entities'], list)

    def test_structure_entite(self, xml_data_center):
        """Chaque entité doit avoir les attributs attendus."""
        panel = xml_data_center['panels'][0]
        entities = panel['struct']['components'][0]['layout']['entities']
        assert len(entities) >= 1, "Aucune entité dans le layout"

        attributs_requis = ['Type', 'Size', 'Color', 'Angle', 'bbox']
        entite = entities[0]
        for attr in attributs_requis:
            assert attr in entite, f"Attribut '{attr}' manquant dans l'entité"

    def test_valeurs_attributs_entite(self, xml_data_center):
        """Les valeurs des attributs doivent être des chaînes convertibles en nombre."""
        panel = xml_data_center['panels'][0]
        entite = panel['struct']['components'][0]['layout']['entities'][0]

        # Vérifier que les attributs numériques sont bien des chaînes de nombres
        for attr in ['Type', 'Size', 'Color', 'Angle']:
            val = entite[attr]
            assert isinstance(val, str), f"{attr} devrait être une chaîne"
            float(val)  # Ne doit pas lever d'exception

    def test_structure_rules(self, xml_data_center):
        """Les règles doivent contenir 'rule_groups' avec des règles."""
        rules = xml_data_center['rules']
        assert 'rule_groups' in rules
        assert isinstance(rules['rule_groups'], list)
        assert len(rules['rule_groups']) >= 1, "Aucun groupe de règles"

    def test_structure_rule_group(self, xml_data_center):
        """Chaque groupe de règles doit contenir 'rules'."""
        rg = xml_data_center['rules']['rule_groups'][0]
        assert 'rules' in rg
        assert isinstance(rg['rules'], list)
        assert len(rg['rules']) >= 1, "Aucune règle dans le groupe"

    def test_structure_rule(self, xml_data_center):
        """Chaque règle doit avoir 'name' et 'attr'."""
        rule = xml_data_center['rules']['rule_groups'][0]['rules'][0]
        assert 'name' in rule, "Règle sans 'name'"
        assert 'attr' in rule, "Règle sans 'attr'"

    def test_coherence_panels(self, xml_data_center):
        """Tous les panneaux doivent avoir la même structure de composants."""
        structures = []
        for panel in xml_data_center['panels']:
            comp_names = [c['name'] for c in panel['struct']['components']]
            structures.append(tuple(comp_names))
        # Tous les panneaux d'un même fichier ont la même structure
        assert len(set(structures)) == 1, (
            f"Les panneaux n'ont pas la même structure de composants : {set(structures)}"
        )

    def test_fichier_inexistant_leve_erreur(self):
        """Un chemin XML inexistant doit lever une erreur."""
        with pytest.raises((FileNotFoundError, OSError)):
            raven_xml_file_parser("/chemin/inexistant/fichier.xml")


# =============================================================================
# TESTS DE raven_xml_file_parser — CONFIGURATIONS MULTI-COMPOSANTS
# =============================================================================

class TestParserMultiComposants:
    """Tests pour les configurations avec plusieurs composants (In/Out)."""

    def test_deux_composants(self, xml_data_in_out):
        """La config in_out doit avoir 2 composants (Out + In)."""
        panel = xml_data_in_out['panels'][0]
        components = panel['struct']['components']
        assert len(components) == 2

    def test_noms_composants_in_out(self, xml_data_in_out):
        """Les composants doivent être nommés 'Out' et 'In'."""
        panel = xml_data_in_out['panels'][0]
        noms = [c['name'] for c in panel['struct']['components']]
        assert 'Out' in noms, f"'Out' manquant, trouvé : {noms}"
        assert 'In' in noms, f"'In' manquant, trouvé : {noms}"

    def test_chaque_composant_a_entites(self, xml_data_in_out):
        """Chaque composant doit avoir au moins une entité."""
        panel = xml_data_in_out['panels'][0]
        for comp in panel['struct']['components']:
            entities = comp['layout']['entities']
            assert len(entities) >= 1, (
                f"Composant '{comp['name']}' sans entité"
            )


# =============================================================================
# TESTS DE build_panel_text_info
# =============================================================================

class TestBuildPanelTextInfo:
    """Tests pour la construction du texte descriptif des panneaux."""

    def test_retourne_string(self, xml_data_center):
        """La fonction doit retourner une chaîne de caractères."""
        result = build_panel_text_info(xml_data_center, 0)
        assert isinstance(result, str)

    def test_contient_struct(self, xml_data_center):
        """Le texte doit commencer par 'Struct:'."""
        result = build_panel_text_info(xml_data_center, 0)
        assert result.startswith("Struct:")

    def test_contient_composant(self, xml_data_center):
        """Le texte doit mentionner les composants."""
        result = build_panel_text_info(xml_data_center, 0)
        assert "Comp1:" in result

    def test_contient_number(self, xml_data_center):
        """Le texte doit contenir la valeur de Number (traduite via NUM_VALUES)."""
        result = build_panel_text_info(xml_data_center, 0)
        assert "Number:" in result

    def test_contient_entites(self, xml_data_center):
        """Le texte doit mentionner au moins une entité."""
        result = build_panel_text_info(xml_data_center, 0)
        assert "E1:" in result

    def test_contient_attributs_entite(self, xml_data_center):
        """Le texte de l'entité doit contenir les attributs clés."""
        result = build_panel_text_info(xml_data_center, 0)
        # Au moins Type et Size doivent apparaître
        assert "Type=" in result
        assert "Size=" in result

    def test_exclut_mask(self, xml_data_center):
        """Le texte ne doit pas contenir 'mask=' (attribut filtré)."""
        result = build_panel_text_info(xml_data_center, 0)
        assert "mask=" not in result.lower() or "mask=" not in result

    def test_exclut_real_bbox(self, xml_data_center):
        """Le texte ne doit pas contenir 'real_bbox=' (attribut filtré)."""
        result = build_panel_text_info(xml_data_center, 0)
        assert "real_bbox=" not in result

    def test_tous_panels_valides(self, xml_data_center):
        """build_panel_text_info doit fonctionner pour les 16 panneaux."""
        for i in range(16):
            result = build_panel_text_info(xml_data_center, i)
            assert isinstance(result, str)
            assert len(result) > 0, f"Panneau {i} : texte vide"

    def test_panels_multi_composants(self, xml_data_in_out):
        """Le texte doit mentionner les deux composants pour la config In/Out."""
        result = build_panel_text_info(xml_data_in_out, 0)
        assert "Comp1:" in result
        assert "Comp2:" in result

    def test_index_hors_limites_leve_erreur(self, xml_data_center):
        """Un index de panneau invalide doit lever une erreur."""
        with pytest.raises(IndexError):
            build_panel_text_info(xml_data_center, 99)

    def test_text_non_identique_entre_panels(self, xml_data_center):
        """Les textes de panneaux différents ne doivent pas être tous identiques."""
        textes = [build_panel_text_info(xml_data_center, i) for i in range(16)]
        tous_identiques = all(t == textes[0] for t in textes[1:])
        # Certains panneaux peuvent être identiques, mais pas tous les 16
        assert not tous_identiques, "Tous les panneaux ont le même texte descriptif"
