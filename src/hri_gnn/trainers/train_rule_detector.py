"""
Training script for the Rule Detector Head (Part 3 of the thesis).

The GNN backbone (RavenGnnModel) is FROZEN. Only the RuleDetectorHead
is trained via supervised learning on rule labels extracted from RAVEN XMLs.

Usage (in notebook):
    from hri_gnn.trainers.train_rule_detector import run_rule_detector_training

    history = run_rule_detector_training(
        dataset_name='raven-20000',
        percentage_to_use=0.75,
        batch_size=256,
        nb_epochs=50,
        learning_rate=1e-3,
    )
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hri_gnn.config.const import GRAPH_NODE_TYPES, RULE_TYPES
from hri_gnn.datasets.rule_dataset import RuleDetectorDataset, rule_detector_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.models.gnn_encoder import RavenGnnModel
from hri_gnn.models.rule_detector import RuleDetectorHead
from hri_gnn.utils import paths


# =============================================================================
# Training
# =============================================================================

def run_rule_detector_training(
    dataset_name: str = 'raven-20000',
    percentage_to_use: float = 0.75,
    batch_size: int = 256,
    nb_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    mlp_hidden: int = 128,
    dropout: float = 0.1,
    gnn_checkpoint: str = 'gnn_model_gine_final.pth',
    session_name: str = 'rule_detector',
):
    """
    Train the RuleDetectorHead on top of a frozen GNN.

    Args:
        gnn_checkpoint: filename of the GNN weights in outputs/models/
        session_name: prefix for saved checkpoint and history files
    """
    output_path = paths.get_output_path()
    paths.output_paths(output_path)

    rule_label_cache = output_path / 'history' / 'rule_labels_all.json'
    if not rule_label_cache.exists():
        raise FileNotFoundError(
            f'Rule label cache not found: {rule_label_cache}\n'
            'Run build_rule_label_cache() first (see datasets/rule_labels.py).'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print(f'\nLoading datasets ({percentage_to_use*100:.0f}% of {dataset_name})...')
    train_ds = RuleDetectorDataset(dataset_name, 'train', rule_label_cache, percentage_to_use)
    val_ds   = RuleDetectorDataset(dataset_name, 'val',   rule_label_cache, percentage_to_use)

    loader_kw = dict(collate_fn=rule_detector_collate, num_workers=4,
                     persistent_workers=True, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # Load frozen GNN (backbone + CNN encoder)
    # ------------------------------------------------------------------
    print(f'\nLoading frozen GNN backbone from {gnn_checkpoint}...')

    cnn_encoder = RavenCnnEncoder()
    cnn_encoder.load_state_dict(
        torch.load(output_path / 'models' / 'raven_cnn_encoder_final.pth', map_location='cpu')
    )
    cnn_encoder.eval()
    for p in cnn_encoder.parameters():
        p.requires_grad_(False)

    gnn_model = RavenGnnModel(GRAPH_NODE_TYPES, hidden_dim=64, num_layers=3, dropout=0.1)
    gnn_model.load_state_dict(
        torch.load(output_path / 'models' / gnn_checkpoint, map_location='cpu')
    )
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    for p in gnn_model.parameters():
        p.requires_grad_(False)

    print(f'  GNN backbone frozen ({sum(p.numel() for p in gnn_model.parameters()):,} params, no grad).')

    # ------------------------------------------------------------------
    # Rule Detector Head (trainable)
    # ------------------------------------------------------------------
    hidden_dim = gnn_model.hidden_dim  # must match GNN
    rule_head = RuleDetectorHead(hidden_dim=hidden_dim, mlp_hidden=mlp_hidden, dropout=dropout)
    rule_head = rule_head.to(device)
    trainable = sum(p.numel() for p in rule_head.parameters() if p.requires_grad)
    print(f'  RuleDetectorHead ({trainable:,} trainable params).')

    optimizer = torch.optim.Adam(rule_head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs, eta_min=1e-5)

    # ------------------------------------------------------------------
    # Pre-compute GNN embeddings once (frozen backbone → cache re-used every epoch)
    # ------------------------------------------------------------------
    n_candidates = len(gnn_model.candidate_ids)

    def _embed_loader(loader, desc):
        """Run the frozen GNN over every batch once and cache the results in RAM."""
        cache = []
        with torch.inference_mode():
            for pyg_batch, batched_layout_keys, _, _, all_comp_ids, labels in tqdm(loader, desc=desc, leave=False):
                pyg_batch = pyg_batch.to(device)
                h_dict = gnn_model.gnn_encoder(
                    {'entity': pyg_batch['entity'].x, 'layout': pyg_batch['layout'].x},
                    {et: pyg_batch[et].edge_index for et in gnn_model.active_edge_types},
                    {et: pyg_batch[et].edge_attr  for et in gnn_model.active_edge_types},
                )
                h_layout     = h_dict['layout'].cpu()          # store on CPU to save VRAM
                n_probs      = pyg_batch.num_graphs // n_candidates
                layout_batch = (pyg_batch['layout'].batch // n_candidates).cpu()

                unique_comp_ids = list(set(all_comp_ids))
                target_comp_id  = unique_comp_ids[0] if len(unique_comp_ids) == 1 else -1

                cache.append((h_layout, batched_layout_keys, layout_batch,
                              n_probs, target_comp_id, labels))
        return cache

    print('Pre-computing GNN embeddings (train)...')
    train_cache = _embed_loader(train_loader, 'Embedding [train]')
    print('Pre-computing GNN embeddings (val)...')
    val_cache   = _embed_loader(val_loader,   'Embedding [val]')
    print(f'  Cached {len(train_cache)} train batches, {len(val_cache)} val batches.\n')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   [],
               'val_type_acc': [], 'val_size_acc': [], 'val_color_acc': [],
               'val_number_position_acc': [], 'val_number_acc': [], 'val_position_acc': []}

    best_val_acc = -1.0
    ckpt_dir = output_path / 'checkpoints' / 'rule_detector'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'Training for {nb_epochs} epochs...\n')

    for epoch in range(nb_epochs):

        # ── Train ─────────────────────────────────────────────────────
        rule_head.train()
        train_loss_sum = 0.0
        train_correct  = 0
        train_total    = 0

        for h_layout, batched_layout_keys, layout_batch, n_probs, target_comp_id, labels in \
                tqdm(train_cache, desc=f'Ep {epoch+1}/{nb_epochs} [train]', leave=False):

            # Move cached CPU tensors to GPU for this step
            h_layout     = h_layout.to(device)
            layout_batch = layout_batch.to(device)
            labels       = labels.to(device)

            logits = rule_head(h_layout, batched_layout_keys, layout_batch, n_probs,
                               target_component_id=target_comp_id)

            losses = rule_head.compute_loss(logits, labels[:n_probs])
            loss   = losses['total_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * n_probs
            accs = rule_head.compute_accuracy(logits, labels[:n_probs])
            train_correct  += accs['mean_acc'] * n_probs
            train_total    += n_probs

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc  = train_correct  / max(train_total, 1)

        # ── Validation ────────────────────────────────────────────────
        rule_head.eval()
        val_loss_sum = 0.0
        val_type_sum = val_size_sum = val_color_sum = 0.0
        val_np_sum   = val_num_sum  = val_pos_sum   = 0.0
        val_total    = 0

        with torch.inference_mode():
            for h_layout, batched_layout_keys, layout_batch, n_probs, target_comp_id, labels in \
                    tqdm(val_cache, desc=f'Ep {epoch+1}/{nb_epochs} [val]', leave=False):

                h_layout     = h_layout.to(device)
                layout_batch = layout_batch.to(device)
                labels       = labels.to(device)

                logits = rule_head(h_layout, batched_layout_keys, layout_batch, n_probs,
                                   target_component_id=target_comp_id)
                losses = rule_head.compute_loss(logits, labels[:n_probs])
                accs   = rule_head.compute_accuracy(logits, labels[:n_probs])

                val_loss_sum += losses['total_loss'].item()          * n_probs
                val_type_sum += accs['type_acc']                      * n_probs
                val_size_sum += accs['size_acc']                      * n_probs
                val_color_sum+= accs['color_acc']                     * n_probs
                val_np_sum   += accs['number_position_acc']           * n_probs
                val_num_sum  += accs['number_acc']                    * n_probs
                val_pos_sum  += accs['position_acc']                  * n_probs
                val_total    += n_probs

        val_loss    = val_loss_sum / max(val_total, 1)
        val_type    = val_type_sum / max(val_total, 1)
        val_size    = val_size_sum / max(val_total, 1)
        val_color   = val_color_sum/ max(val_total, 1)
        val_np      = val_np_sum   / max(val_total, 1)
        val_num     = val_num_sum  / max(val_total, 1)
        val_pos     = val_pos_sum  / max(val_total, 1)
        # mean over the 6 heads (spatial heads will be 0 when absent — weighted fairly)
        val_mean    = (val_type + val_size + val_color + val_np + val_num + val_pos) / 6

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        best_marker = ''
        if val_mean > best_val_acc:
            best_val_acc = val_mean
            best_path = ckpt_dir / f'{session_name}_best.pth'
            torch.save(rule_head.state_dict(), best_path)
            best_marker = ' ★'

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_mean)
        history['val_type_acc'].append(val_type)
        history['val_size_acc'].append(val_size)
        history['val_color_acc'].append(val_color)
        history['val_number_position_acc'].append(val_np)
        history['val_number_acc'].append(val_num)
        history['val_position_acc'].append(val_pos)

        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1:3d}/{nb_epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} '
              f'T:{val_type:.3f} S:{val_size:.3f} C:{val_color:.3f} '
              f'NP:{val_np:.3f} N:{val_num:.3f} P:{val_pos:.3f} '
              f'Mean:{val_mean:.4f} | LR: {lr:.2e}{best_marker}')

    # ------------------------------------------------------------------
    # Save final model + history
    # ------------------------------------------------------------------
    final_path = output_path / 'models' / f'{session_name}_final.pth'
    torch.save(rule_head.state_dict(), final_path)
    print(f'\n[OK] Final model saved: {final_path}')
    print(f'[OK] Best model saved : {best_path}  (val mean acc = {best_val_acc:.4f})')

    history_path = output_path / 'history' / 'rule_detector' / f'{session_name}_history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[OK] History saved    : {history_path}')

    return history, rule_head
