"""
Rule Detector — Part 3 of the thesis.

Architecture:
  [Frozen GNN] → layout node embeddings (hidden_dim per node)
       ↓
  [Context Pooling] — aggregate the 8 context panel embeddings into one vector
       ↓
  [RuleDetectorHead] — 3 independent classifiers (type / size / color)
       ↓
  Output: 3 × 4 logits  (one 4-class distribution per attribute)


Training:
  - The GNN backbone is FROZEN (no gradient updates)
  - Only the RuleDetectorHead parameters are trained
  - Supervision comes from the ground-truth rule labels in the RAVEN XML files
  - Loss: CrossEntropy on each of the 3 heads, summed

Usage (after training):
  predictions = rule_detector.predict(pyg_batch, batched_layout_keys, n_problems)
  # → list of dicts: [{'type': 0, 'size': 1, 'color': 0}, ...]
  # which means: [{'type': Constant, 'size': Progression, 'color': Constant}, ...]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from hri_gnn.config.const import RULE_TYPES


# ── Context pooling ───────────────────────────────────────────────────────────

class ContextPanelPooling(nn.Module):
    """
    Pools the layout node embeddings of the 8 CONTEXT panels (panel_ids 0–7)
    into a single fixed-size vector per problem.

    For multi-component configs, restrict pooling to a single component by
    passing target_component_id. This gives the model a per-component view
    so it can detect the rules of the inner entity separately from the outer.

    Strategy: mean-pooling over the selected layout nodes.
    """

    def forward(
        self,
        h_layout: Tensor,                  # (N_layout_total, hidden_dim)
        batched_layout_keys: list[tuple],  # list of (panel_id, component_id) per layout node
        layout_batch: Tensor,              # (N_layout_total,) — graph index per node
        n_problems: int,
        hidden_dim: int,
        device: torch.device,
        target_component_id: int = -1,     # -1 = all components; ≥0 = specific component only
    ) -> Tensor:
        """
        Args:
            target_component_id: if ≥ 0, only pool layout nodes whose component_id
                                  matches this value (AND panel_id < 8).
                                  -1 pools all components (single-component configs).

        Returns: context_embeddings  (n_problems, hidden_dim)
        """
        # Build integer tensors from the key list on CPU, then move to device in one transfer.
        # This avoids a per-element Python loop inside the forward pass which stalls CUDA
        # long enough to trigger Windows TDR (cudaErrorLaunchTimeout).
        keys_tensor = torch.tensor(batched_layout_keys, dtype=torch.long)  # (N, 2) on CPU
        panel_ids = keys_tensor[:, 0].to(device)   # (N,)
        comp_ids  = keys_tensor[:, 1].to(device)   # (N,)

        if target_component_id < 0:
            # Legacy / single-component: pool all context panels regardless of component
            context_mask = panel_ids < 8
        else:
            # Per-component mode: only pool the target component's layout nodes
            context_mask = (panel_ids < 8) & (comp_ids == target_component_id)

        h_context     = h_layout[context_mask]       # (N_context, hidden_dim)
        batch_context = layout_batch[context_mask]   # (N_context,)

        context_emb = torch.zeros(n_problems, hidden_dim, device=device)
        counts      = torch.zeros(n_problems, 1, device=device)
        context_emb.scatter_add_(0, batch_context.unsqueeze(1).expand_as(h_context), h_context)
        # Reuse a pre-allocated ones tensor to avoid repeated allocation
        counts.scatter_add_(0, batch_context.unsqueeze(1),
                            torch.ones(h_context.shape[0], 1, device=device))
        counts = counts.clamp(min=1.0)
        context_emb = context_emb / counts

        return context_emb  # (n_problems, hidden_dim)


# ── Rule Detector Head ────────────────────────────────────────────────────────

class RuleDetectorHead(nn.Module):
    """
    Lightweight MLP head that predicts which rule governs each of 6 attributes
    (Type, Size, Color, Number/Position, Number, Position) given the pooled
    context panel embedding from a frozen GNN.

    For each attribute, a 4-class classifier is trained:
        0 = Constant  |  1 = Progression  |  2 = Arithmetic  |  3 = Distribute_Three

    Attributes not present for a given problem (label == -1) are masked out
    in the loss and accuracy computation — the head is only trained when
    the corresponding rule actually appears in the RAVEN XML.
    """

    NUM_RULES  = 4
    ATTR_NAMES = ['type', 'size', 'color', 'number_position', 'number', 'position']

    def __init__(self, hidden_dim: int, mlp_hidden: int = 128, dropout: float = 0.1):
        """
        Args:
            hidden_dim: output dimension of the frozen GNN encoder (= GNN hidden_dim)
            mlp_hidden: hidden size of the rule classifier MLP
            dropout: dropout probability inside the MLP
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_pooling = ContextPanelPooling()

        # Shared trunk: compress the pooled embedding
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Six independent classifiers — one per attribute
        # Always present:
        self.type_head             = nn.Linear(mlp_hidden, self.NUM_RULES)
        self.size_head             = nn.Linear(mlp_hidden, self.NUM_RULES)
        self.color_head            = nn.Linear(mlp_hidden, self.NUM_RULES)
        # Spatial heads — each trained only when that form is present in the XML:
        self.number_position_head  = nn.Linear(mlp_hidden, self.NUM_RULES)  # Number/Position combined
        self.number_head           = nn.Linear(mlp_hidden, self.NUM_RULES)  # Number alone
        self.position_head         = nn.Linear(mlp_hidden, self.NUM_RULES)  # Position alone

    def forward(
        self,
        h_layout: Tensor,
        batched_layout_keys: list[tuple],
        layout_batch: Tensor,
        n_problems: int,
        target_component_id: int = -1,
    ) -> dict[str, Tensor]:
        """
        Args:
            h_layout: layout node embeddings from the frozen GNN  (N_layout, hidden_dim)
            batched_layout_keys: list of (panel_id, component_id) for each layout node
            layout_batch: graph index per layout node  (N_layout,)
            n_problems: number of problems in this batch
            target_component_id: which component to pool (-1 = all, 0 = primary, 1 = secondary)

        Returns:
            dict with keys 'type', 'size', 'color',
            'number_position', 'number', 'position',
            each a Tensor of shape (n_problems, 4)
        """
        device = h_layout.device

        # 1. Pool context panel embeddings → (n_problems, hidden_dim)
        context_emb = self.context_pooling(
            h_layout, batched_layout_keys, layout_batch,
            n_problems, self.hidden_dim, device,
            target_component_id=target_component_id,
        )

        # 2. Shared trunk
        z = self.trunk(context_emb)  # (n_problems, mlp_hidden)

        # 3. Per-attribute classification (all 6 heads always computed)
        return {
            'type':            self.type_head(z),            # always present
            'size':            self.size_head(z),            # always present
            'color':           self.color_head(z),           # always present
            'number_position': self.number_position_head(z), # masked in loss when absent
            'number':          self.number_head(z),          # masked in loss when absent
            'position':        self.position_head(z),        # masked in loss when absent
        }

    def predict(
        self,
        h_layout: Tensor,
        batched_layout_keys: list[tuple],
        layout_batch: Tensor,
        n_problems: int,
        target_component_id: int = -1,
    ) -> list[dict[str, str]]:
        """
        Convenience method: returns human-readable rule predictions.

        Returns:
            list of dicts, one per problem:
            [{'type': 'Constant', 'size': 'Progression', 'color': 'Constant'}, ...]
        """
        with torch.inference_mode():
            logits = self.forward(h_layout, batched_layout_keys, layout_batch, n_problems,
                                  target_component_id=target_component_id)

        results = []
        for i in range(n_problems):
            results.append({
                'type':            RULE_TYPES[logits['type'][i].argmax().item()],
                'size':            RULE_TYPES[logits['size'][i].argmax().item()],
                'color':           RULE_TYPES[logits['color'][i].argmax().item()],
                'number_position': RULE_TYPES[logits['number_position'][i].argmax().item()],
                'number':          RULE_TYPES[logits['number'][i].argmax().item()],
                'position':        RULE_TYPES[logits['position'][i].argmax().item()],
            })
        return results

    def compute_loss(
        self,
        logits: dict[str, Tensor],
        labels: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute masked cross-entropy loss for each attribute head.

        Args:
            logits: output of forward() — dict with 6 keys,
                    each a Tensor (n_problems, 4)
            labels: (n_problems, 6) ground-truth indices
                    columns: [type, size, color, number_position, number, position]
                    -1 means the attribute is absent for that problem — skipped in loss.

        Returns:
            dict with per-head losses and 'total_loss' (sum of present heads only).
            Absent heads contribute 0.0 to total_loss.
        """
        def masked_ce(logit, lbl_col):
            """Cross-entropy only on samples where lbl_col != -1."""
            mask = lbl_col >= 0
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logit.device)
            return F.cross_entropy(logit[mask], lbl_col[mask])

        type_loss             = masked_ce(logits['type'],            labels[:, 0])
        size_loss             = masked_ce(logits['size'],            labels[:, 1])
        color_loss            = masked_ce(logits['color'],           labels[:, 2])
        number_position_loss  = masked_ce(logits['number_position'], labels[:, 3])
        number_loss           = masked_ce(logits['number'],          labels[:, 4])
        position_loss         = masked_ce(logits['position'],        labels[:, 5])

        total_loss = (type_loss + size_loss + color_loss
                      + number_position_loss + number_loss + position_loss)

        return {
            'type_loss':            type_loss,
            'size_loss':            size_loss,
            'color_loss':           color_loss,
            'number_position_loss': number_position_loss,
            'number_loss':          number_loss,
            'position_loss':        position_loss,
            'total_loss':           total_loss,
        }

    def compute_accuracy(
        self,
        logits: dict[str, Tensor],
        labels: Tensor,
    ) -> dict[str, float]:
        """
        Compute per-attribute accuracy, skipping absent attributes (label == -1).

        Args:
            logits: output of forward()
            labels: (n_problems, 6) ground-truth indices

        Returns:
            dict with per-attribute accuracies and 'mean_acc'
            (averaged only over attributes present in this batch).
        """
        def masked_acc(logit, lbl_col):
            """Accuracy only on samples where lbl_col != -1. Returns (acc, n_present)."""
            mask = lbl_col >= 0
            n = mask.sum().item()
            if n == 0:
                return 0.0, 0
            acc = (logit[mask].argmax(dim=1) == lbl_col[mask]).float().mean().item()
            return acc, n

        type_acc,  nt  = masked_acc(logits['type'],            labels[:, 0])
        size_acc,  ns  = masked_acc(logits['size'],            labels[:, 1])
        color_acc, nc  = masked_acc(logits['color'],           labels[:, 2])
        np_acc,    nnp = masked_acc(logits['number_position'], labels[:, 3])
        num_acc,   nn_ = masked_acc(logits['number'],          labels[:, 4])
        pos_acc,   npo = masked_acc(logits['position'],        labels[:, 5])

        # Mean accuracy over heads that are active in this batch
        active_accs  = [a for a, n in [(type_acc, nt), (size_acc, ns), (color_acc, nc),
                                        (np_acc, nnp), (num_acc, nn_), (pos_acc, npo)] if n > 0]
        mean_acc = sum(active_accs) / len(active_accs) if active_accs else 0.0

        return {
            'type_acc':            type_acc,
            'size_acc':            size_acc,
            'color_acc':           color_acc,
            'number_position_acc': np_acc,
            'number_acc':          num_acc,
            'position_acc':        pos_acc,
            'mean_acc':            mean_acc,
        }
