import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hri_gnn.datasets.precomputed import PrecomputedRavenDataset, precomputed_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.models.gnn_encoder import RavenGnnModel
from hri_gnn.trainers import GnnTrainer
from hri_gnn.metrics import TrainingTimer
from hri_gnn.utils import paths
from hri_gnn.config.const import GRAPH_NODE_TYPES


# =============================================================================
# Helpers
# =============================================================================

def print_validation_results(validation_results, save_path=None):
    """
    Prints the evaluation metrics in a formatted way.

    Args:
        validation_results (dict): Dictionary containing the validation results.
        save_path (Path, optional): Path to save the validation results. Defaults to None.
    """
    print("\n" + "=" * 60)
    print("  RESULTS ON VALIDATION SET  ")
    print(f" Number: \n  - Problems : {validation_results['val_num_problems']} \n  - Batches  : {validation_results['val_num_batches']}")
    print("=" * 60)
    print(f"  Accuracy  : {validation_results['val_accuracy']*100:.2f}%")
    print(f"  Precision : {validation_results['val_precision']:.4f}")
    print(f"  Recall    : {validation_results['val_recall']:.4f}")
    print(f"  F1 score  : {validation_results['val_f1']:.4f}")

    if save_path:
        validation_results_path = save_path / "validation_results.json"
        with open(validation_results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n[OK] Validation results saved : {validation_results_path}")


def find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> tuple[Path | None, int]:
    """
    Scans checkpoint_dir for the latest checkpoint matching prefix.

    Returns:
        (path, epoch) of the most recent checkpoint, or (None, 0) if none found.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_checkpoint_epoch_(\d+)\.pth")
    best_path, best_epoch = None, 0
    if not checkpoint_dir.exists():
        return None, 0
    for f in checkpoint_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = f
    return best_path, best_epoch


def load_history(history_path: Path) -> dict:
    """Loads a previously saved training history JSON, or returns an empty dict."""
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return {}


# =============================================================================
# Training Process
# =============================================================================

def run_training(dataset_name='raven-20000',
                 percentage_to_use=0.5,
                 batch_size=256,
                 nb_epochs=150,
                 check_frequency=10,
                 hidden_dim=64,
                 num_layers=3,
                 dropout=0.1,
                 learning_rate=5e-4,
                 weight_decay=1e-4,
                 mode='min',
                 factor=0.5,
                 patience=10,
                 min_lr=1e-5,
                 session_name='session',
                 resume=True,
                 ):
    """
    Train the GNN model with automatic checkpoint resume support.

    Args:
        resume (bool): If True, automatically detects the latest checkpoint and
                       continues training from that epoch. If False, always trains
                       from scratch.
        session_name (str): Prefix used for checkpoint filenames and history file.
                            Use a unique name per experiment so sessions don't
                            overwrite each other.
    """
    output_path = paths.get_output_path()
    paths.output_paths(output_path)

    checkpoint_dir = output_path / "checkpoints" / "gnn"
    history_path = output_path / "history" / "gnn" / f"training_history_{session_name}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f" GPU : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    print(f"\nLoading dataset : {dataset_name}")
    print(f"  Percentage to use : {percentage_to_use * 100}%")

    train_dataset = PrecomputedRavenDataset(dataset=dataset_name, split='train',
                                            percentage_limit=percentage_to_use, configs=None)
    val_dataset   = PrecomputedRavenDataset(dataset=dataset_name, split='val',
                                            percentage_limit=percentage_to_use, configs=None)
    test_dataset  = PrecomputedRavenDataset(dataset=dataset_name, split='test',
                                            percentage_limit=percentage_to_use, configs=None)

    print(f"\nBatch size : {batch_size}")
    print(f"Train dataset loaded : {len(train_dataset)} problems")
    print(f"Val dataset loaded   : {len(val_dataset)} problems")
    print(f"Test dataset loaded  : {len(test_dataset)} problems")

    loader_kwargs = dict(collate_fn=precomputed_collate, num_workers=4,
                         persistent_workers=True, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    raven_encoder = RavenCnnEncoder()
    raven_encoder.load_state_dict(torch.load(f=output_path / 'models' / 'raven_cnn_encoder_final.pth',
                                             map_location='cpu'))

    gnn_model = RavenGnnModel(GRAPH_NODE_TYPES, hidden_dim=hidden_dim,
                              num_layers=num_layers, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    history = {}

    if resume:
        ckpt_path, last_epoch = find_latest_checkpoint(checkpoint_dir, f"gnn_model_{session_name}")
        if ckpt_path is not None:
            print(f"\n[RESUME] Found checkpoint at epoch {last_epoch}: {ckpt_path}")
            gnn_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = last_epoch
            history = load_history(history_path)
            remaining_epochs = nb_epochs - start_epoch
            if remaining_epochs <= 0:
                print(f"[RESUME] Training already completed ({last_epoch}/{nb_epochs} epochs). Nothing to do.")
                return history, None, 0
            print(f"[RESUME] Resuming from epoch {start_epoch}. Remaining: {remaining_epochs} epochs.")
        else:
            print("\n[RESUME] No checkpoint found. Starting from scratch.")
    else:
        print("\nStarting training from scratch.")

    remaining_epochs = nb_epochs - start_epoch

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    gnn_trainer = GnnTrainer(gnn_model, raven_encoder, optimizer, scheduler,
                             metrics=None, node_types=None, active_edge_types=None,
                             include_columns=True, device=device, output_path=output_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    with TrainingTimer(model=gnn_model) as train_timer:
        new_history = gnn_trainer.fit(
            train_dataloader, test_dataloader,
            epochs=remaining_epochs,
            max_grad_norm=1.0,
            checkpoint_name_prefix=session_name,
            check_frequency=check_frequency,
            save_final_version=True,
            verbose=True,
            history_save_path=output_path / "history" / "gnn" / f"training_history_{session_name}_partial.json",
        )
    model_training_time = train_timer.elapsed

    # Merge old history (from previous sessions) with new history
    for key, value in new_history.items():
        if key in history and isinstance(value, list) and isinstance(history.get(key), list):
            history[key] = history[key] + value   # concatenate epoch lists
        else:
            history[key] = value                  # scalars / matrices: keep latest

    # ------------------------------------------------------------------
    # Validation on val split
    # ------------------------------------------------------------------
    validation_results = gnn_trainer.validate(val_dataloader)
    validation_result_save_path = output_path / "history" / "gnn"
    print_validation_results(validation_results, save_path=validation_result_save_path)

    # ------------------------------------------------------------------
    # Save merged history
    # ------------------------------------------------------------------
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved : {history_path}")

    return history, validation_results, model_training_time