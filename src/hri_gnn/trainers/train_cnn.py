"""
Main script to train the cnn encoder model for the RAVEN dataset.

Usage:
    python -m hri_gnn.trainers.train_cnn
"""
import json

import torch
from torch.utils.data import DataLoader

from hri_gnn.datasets import RavenDataset , raven_flat_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.losses import CompositeLoss
from hri_gnn.metrics import CompositeMetric, TrainingTimer
from hri_gnn.trainers.cnn_trainer import CnnTrainer
from hri_gnn.config.const import METRICS

from hri_gnn.visualization import oracle

from hri_gnn.utils import paths


def print_validation_results(validation_results, save_path=None):
    """
    Prints the evaluation metrics in a formatted way.

    Args:
        validation_results (dict): Dictionary containing the validation results.
        save_path (Path, optional): Path to save the validation results. Defaults to None.
    """
    
    # Print validation results
    print("\n" + "=" * 60)
    print("  RESULTS ON VALIDATION SET  ")
    print(f" Number: \n  - Crops   : {validation_results['val_num_of_crops']} \n  - Batches : {validation_results['val_num_of_batches']}")
    print("=" * 60)
    print(f"  Loss   : {validation_results['val_loss']:.4f}")
    print(f"  Accuracy : {validation_results['val_accuracy']*100:.2f}%")
    print(f"  Precision           : {validation_results['val_precision']:.4f}")
    print(f"  Recall           : {validation_results['val_recall']:.4f}")
    print(f"  F1 score          : {validation_results['val_f1']:.4f}")
    print("-" * 14 + " DETAILED LOSSES " + "-" * 15)
    print(f"  Type Loss    : {validation_results['val_loss_type']:.4f}")
    print(f"  Size Loss    : {validation_results['val_loss_size']:.4f}")
    print(f"  Color Loss   : {validation_results['val_loss_color']:.4f}")
    print(f"  Angle Loss   : {validation_results['val_loss_angle']:.4f}")
    print("-" * 13 + " DETAILED ACCURACIES " + "-" * 13)
    print(f"  Type Accuracy : {validation_results['val_accuracy_type']*100:.2f}%")
    print(f"  Size Accuracy : {validation_results['val_accuracy_size']*100:.2f}%")
    print(f"  Color Accuracy : {validation_results['val_accuracy_color']*100:.2f}%")
    print(f"  Angle Accuracy : {validation_results['val_accuracy_angle']*100:.2f}%")
    print("-" * 14 + " DETAILED PRECISIONS " + "-" * 13)
    print(f"  Type Precision     : {validation_results['val_precision_type']:.4f}")
    print(f"  Size Precision     : {validation_results['val_precision_size']:.4f}")
    print(f"  Color Precision    : {validation_results['val_precision_color']:.4f}")
    print(f"  Angle Precision    : {validation_results['val_precision_angle']:.4f}")
    print("-" * 14 + " DETAILED RECALLS " + "-" * 15)
    print(f"  Type Recall     : {validation_results['val_recall_type']:.4f}")
    print(f"  Size Recall     : {validation_results['val_recall_size']:.4f}")
    print(f"  Color Recall    : {validation_results['val_recall_color']:.4f}")
    print(f"  Angle Recall    : {validation_results['val_recall_angle']:.4f}")
    print("-" * 13 + " DETAILED F1 SCORES " + "-" * 14)
    print(f"  Type F1 score     : {validation_results['val_f1_type']:.4f}")
    print(f"  Size F1 score     : {validation_results['val_f1_size']:.4f}")
    print(f"  Color F1 score    : {validation_results['val_f1_color']:.4f}")
    print(f"  Angle F1 score    : {validation_results['val_f1_angle']:.4f}")
    
    # Saving validation results
    if save_path:
        validation_results_path = save_path/"validation_results.json"
        with open(validation_results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n[OK] Validation results saved : {validation_results_path}")


# =============================================================================
# Training Process
# =============================================================================

def run_training(dataset_name, percentage_to_use, nb_epochs, check_frequency, batch_size, learning_rate):

    output_path = paths.get_output_path()
    paths.output_paths(output_path)
    
    """Configure and run the training process for the CNN encoder model on the RAVEN dataset."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # 1. Load datasets

    print(f"\nLoading dataset : {dataset_name}")
    print(f"  Percentage to use : {percentage_to_use*100}%")

    train_dataset = RavenDataset(
                        dataset=dataset_name,
                        split='train',
                        percentage_limit=percentage_to_use)
    val_dataset = RavenDataset(
                        dataset=dataset_name,
                        split='val',
                        percentage_limit=percentage_to_use)
    test_dataset = RavenDataset(
                        dataset=dataset_name,
                        split='test',
                        percentage_limit=percentage_to_use)

    # 2. DataLoaders

    print(f"\nBatch size : {batch_size}")
    print(f"Train dataset loaded : {len(train_dataset)} problems")
    print(f"Val dataset loaded : {len(val_dataset)} problems")
    print(f"Test dataset loaded : {len(test_dataset)} problems")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=raven_flat_collate)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=raven_flat_collate)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=raven_flat_collate)

    # 3. Model
    model = RavenCnnEncoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModele : RavenCnnEncoder ({total_params:,} parametres)")
    
    # 4. Optimizer and Loss

    print(f"\nOptimizer : Adam")
    print(f" Learning rate : {learning_rate}")
    print(f" Loss : CompositeLoss ")
    print(f" Metrics : CompositeMetric")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CompositeLoss()
    metrics  = CompositeMetric(device=device)

    # 5. Training
    trainer = CnnTrainer(model=model, optimizer=optimizer, criterion=criterion, metrics=metrics, device=device)
    print(f"epochs={nb_epochs}, batch_size={batch_size}, lr={learning_rate}")

    with TrainingTimer(model=model) as train_timer:
        history = trainer.fit(train_loader, test_loader, epochs=nb_epochs, checkpoint_dir=output_path, check_frequency=check_frequency)
    model_training_time = train_timer.elapsed

    # 6. Validation
    validation_results = trainer.validate(val_loader)
    print_validation_results(validation_results,save_path=output_path)

    # 7. Saving the training history
    history_path = output_path/"training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved : {history_path}")

    # 8. Learning curves
    for metric in METRICS:
        if metric == "confusion_matrix":
            oracle.plot_confusion_matrices(history, validation_results, save_path=output_path)
        else:
            oracle.plot_metrics_curves(history, metric, nb_epochs, save_path=output_path)
            oracle.plot_composed_metrics_curves(history, metric, nb_epochs, save_path=output_path)
    
    oracle.plot_attributes_curves(history, nb_epochs, save_path=output_path)
    

    # 9. Saving the final model
    model_path = output_path/"models"/"raven_cnn_encoder_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved : {model_path}")


    return history, validation_results ,model_training_time


# =============================================================================
# ENTRY POINT
# =============================================================================

# if __name__ == "__main__":
#     run_training()