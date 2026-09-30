"""
Rule label extraction from RAVEN XML files.

The RAVEN dataset encodes the ground-truth abstract rules that govern each problem
directly in the XML file under the <Rules> → <Rule_Group> → <Rule> tags.

Each Rule_Group represents ONE component (layout) of the problem.
- Single-component configs (center_single, distribute_four, distribute_nine):
    → 1 Rule_Group → 1 set of rules
- Two-component configs (in_center_single_out_center_single, left_*, up_*, in_distribute_*):
    → 2 Rule_Groups → 2 independent sets of rules (one per layout component)

Each rule has a 'name' which is one of:
  Constant | Progression | Arithmetic | Distribute_Three

The spatial attribute can appear as:
  - 'Number/Position'  (combined)
  - 'Number' alone
  - 'Position' alone
  - Both 'Number' AND 'Position' separately

This module provides:
  - RuleLabel: labels for ONE component of ONE problem
  - parse_rules_from_xml(): returns list[RuleLabel] — one per Rule_Group
  - build_rule_label_cache(): scan entire dataset, key = (problem_id, component_id)
  - load_rule_label_cache(): reload from JSON
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from hri_gnn.config.const import RULE_TYPES, RULE_ATTRIBUTES


# ── Constants ─────────────────────────────────────────────────────────────────

# Map rule name → integer index  (matches RULE_TYPES order in const.py)
RULE_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(RULE_TYPES)}
# e.g. {'Constant': 0, 'Progression': 1, 'Arithmetic': 2, 'Distribute_Three': 3}

# Spatial attribute names — any of these maps to the spatial category
SPATIAL_ATTRS = {'Number/Position', 'Number', 'Position'}

# Entity-level attributes we care about for the Rule Detector
ENTITY_RULE_ATTRS = ["Type", "Size", "Color"]


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RuleLabel:
    """
    Ground-truth rule type labels for ONE COMPONENT of one RAVEN problem.

    A single problem can have multiple components (Rule_Groups), each with
    its own independent set of rules. This class stores the rules for ONE
    component. The component_id matches the Rule_Group id in the XML.

    Attributes are assigned one of 4 rule types:
        0 = Constant
        1 = Progression
        2 = Arithmetic
        3 = Distribute_Three

    Spatial attributes can appear in 3 forms in the XML:
        - 'Number/Position'  → number_position_rule set, number_rule = position_rule = -1
        - 'Number' only      → number_rule set, position_rule = -1, number_position_rule = -1
        - 'Position' only    → position_rule set, number_rule = -1, number_position_rule = -1
        - Both separately    → both number_rule and position_rule set

    Use spatial_rule() to get the best available spatial label for training.
    """
    problem_id:    int
    component_id:  int   # Rule_Group id (0 = first/only component, 1 = second, ...)
    config:        str
    split:         str

    type_rule:             int   # rule governing the Type attribute
    size_rule:             int   # rule governing the Size attribute
    color_rule:            int   # rule governing the Color attribute
    number_position_rule:  int   # combined Number/Position rule (-1 if split)
    number_rule:           int   # Number-only rule (-1 if not present)
    position_rule:         int   # Position-only rule (-1 if not present)

    # def spatial_rule(self) -> int:
    #     """
    #     Returns the best available spatial rule label (for backward compat / analysis).
    #     Prefer using as_tensor_labels() which returns all 6 labels independently.
    #     Priority: number_position_rule → number_rule → position_rule.
    #     """
    #     if self.number_position_rule >= 0:
    #         return self.number_position_rule
    #     if self.number_rule >= 0:
    #         return self.number_rule
    #     return self.position_rule

    def as_tensor_labels(self) -> list[int]:
        """
        Returns all 6 rule labels as a list of ints.
        -1 means the attribute is not present for this problem/component.

        Order: [type_rule, size_rule, color_rule,
                number_position_rule, number_rule, position_rule]

        Examples:
          center_single    → [t, s, c, np, -1, -1]   (always Number/Position)
          distribute_nine  → [t, s, c, -1,  n,  p]   (separate Number + Position)
          distribute_four  → [t, s, c, np, -1, -1]   (or [t,s,c,-1,n,p] depending on problem)
        """
        return [
            self.type_rule,
            self.size_rule,
            self.color_rule,
            self.number_position_rule,   # -1 if not present
            self.number_rule,            # -1 if not present
            self.position_rule,          # -1 if not present
        ]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_rules_from_xml(
    xml_path: Path,
    problem_id: int,
    config: str,
    split: str,
) -> list[RuleLabel]:
    """
    Parse the rule labels for ALL components of one RAVEN problem.

    Returns one RuleLabel per Rule_Group (per component).
    - Single-component configs: returns list of length 1
    - Two-component configs:    returns list of length 2

    Returns empty list if the XML is malformed or has no valid rules.

    Args:
        xml_path: path to the .xml file
        problem_id: integer ID of the problem (from filename)
        config: configuration name (e.g. 'in_center_single_out_center_single')
        split: 'train', 'val', or 'test'
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    rules_elem = root.find('Rules')
    if rules_elem is None:
        return []

    results: list[RuleLabel] = []

    for rule_group in rules_elem.findall('Rule_Group'):
        component_id = int(rule_group.attrib.get('id', 0))

        # Build rule map for this component
        rule_map: dict[str, int] = {}
        for rule_elem in rule_group:
            attr = rule_elem.attrib.get('attr', '')
            name = rule_elem.attrib.get('name', '')
            idx  = RULE_NAME_TO_IDX.get(name, -1)
            if idx >= 0:
                rule_map[attr] = idx

        # Type, Size, Color are required for each component
        if not all(a in rule_map for a in ['Type', 'Size', 'Color']):
            continue   # skip malformed Rule_Group, try next

        # Spatial: may be combined or split
        number_position_rule = rule_map.get('Number/Position', -1)
        number_rule          = rule_map.get('Number', -1)
        position_rule        = rule_map.get('Position', -1)

        # Must have at least one spatial label
        if number_position_rule < 0 and number_rule < 0 and position_rule < 0:
            continue

        results.append(RuleLabel(
            problem_id=problem_id,
            component_id=component_id,
            config=config,
            split=split,
            type_rule=rule_map['Type'],
            size_rule=rule_map['Size'],
            color_rule=rule_map['Color'],
            number_position_rule=number_position_rule,
            number_rule=number_rule,
            position_rule=position_rule,
        ))

    return results


# ── Cache builder ─────────────────────────────────────────────────────────────

# Cache key type: (config_id, problem_id, component_id)
# config_id is the integer index in Configuration.all() (same as in PrecomputedRavenDataset)
LabelKey = tuple[int, int, int]

def build_rule_label_cache(
    raw_data_path: Path,
    dataset_name: str = 'raven-20000',
    configs: Optional[list[str]] = None,
    splits: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> dict[str, dict[LabelKey, RuleLabel]]:
    """
    Scan the entire dataset and extract rule labels for all problems and components.

    Returns a nested dict:
        labels[split][(config_id, problem_id, component_id)] = RuleLabel

    The 3-part key (config_id, problem_id, component_id) is the only truly unique
    identifier: problem_id alone is NOT unique across configs (problem 0 in
    center_single is a different problem from problem 0 in distribute_four).

    Args:
        raw_data_path: path to the raw data folder
        dataset_name: e.g. 'raven-20000'
        configs: list of config names (None = all 7)
        splits: list of splits (None = train + val + test)
        save_path: if provided, save the cache as JSON here

    Example:
        labels = build_rule_label_cache(raw_data_path, 'raven-20000')
        # center_single (config_id=0), problem 1234, component 0:
        lbl = labels['train'][(0, 1234, 0)]
        # in_center_single_out_center_single (config_id=3), problem 42:
        inner = labels['train'][(3, 42, 0)]
        outer = labels['train'][(3, 42, 1)]
    """
    from hri_gnn.config.const import Configuration

    if configs is None:
        configs = Configuration.all()
    if splits is None:
        splits = ['train', 'val', 'test']

    dataset_path = raw_data_path / dataset_name
    labels: dict[str, dict[LabelKey, RuleLabel]] = {split: {} for split in splits}

    total = 0
    errors = 0

    all_configs = Configuration.all()

    for config in configs:
        config_id   = all_configs.index(config)
        config_path = dataset_path / config
        if not config_path.exists():
            print(f'  [WARN] Config path not found: {config_path}')
            continue

        for xml_file in sorted(config_path.glob('RAVEN_*.xml')):
            stem = xml_file.stem   # e.g. 'RAVEN_13869_test'
            parts = stem.split('_')
            if len(parts) < 3:
                continue
            try:
                problem_id = int(parts[1])
            except ValueError:
                continue
            file_split = parts[2]  # 'train', 'val', or 'test'

            if file_split not in splits:
                continue

            component_labels = parse_rules_from_xml(xml_file, problem_id, config, file_split)
            if not component_labels:
                errors += 1
                continue

            for lbl in component_labels:
                key = (config_id, lbl.problem_id, lbl.component_id)
                labels[file_split][key] = lbl   # always overwrite — same file, exact match
                total += 1

    # Count unique (config_id, problem_id) pairs and total component-labels
    n_unique  = {split: len(set((k[0], k[1]) for k in labels[split])) for split in splits}
    n_total_k = {split: len(labels[split]) for split in splits}

    print(f'Rule label cache built: {total} component-labels, {errors} XML errors.')
    for split in splits:
        print(f'  {split}: {n_unique[split]} unique (config, problem) pairs, '
              f'{n_total_k[split]} component-labels '
              f'(avg {n_total_k[split]/max(n_unique[split],1):.2f} components/problem)')

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            split: {
                f'{cfg_id}_{pid}_{cid}': lbl.to_dict()
                for (cfg_id, pid, cid), lbl in split_labels.items()
            }
            for split, split_labels in labels.items()
        }
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f'  Saved to: {save_path}')

    return labels


def load_rule_label_cache(cache_path: Path) -> dict[str, dict[LabelKey, RuleLabel]]:
    """
    Load a previously saved rule label cache from a JSON file.

    Returns:
        labels[split][(config_id, problem_id, component_id)] = RuleLabel
    """
    with open(cache_path, 'r') as f:
        raw = json.load(f)

    labels: dict[str, dict[LabelKey, RuleLabel]] = {}
    for split, split_data in raw.items():
        labels[split] = {}
        for key_str, d in split_data.items():
            cfg_str, pid_str, cid_str = key_str.split('_')
            key = (int(cfg_str), int(pid_str), int(cid_str))
            labels[split][key] = RuleLabel(**d)
    return labels
