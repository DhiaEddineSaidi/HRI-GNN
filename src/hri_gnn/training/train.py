"""
Main script to train the cnn encoder model for the RAVEN dataset.

Usage:
    python -m training.train
"""
from hri_gnn.config.const import ENTITY_ATTRIBUTES
import os 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json

import torch
from torch.utils.data import DataLoader

from hri_gnn.datasets import RavenDataset , raven_flat_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.losses import CompositeLoss
from hri_gnn.metrics import CompositeMetric, TrainingTimer
from hri_gnn.training.trainer import Trainer
from hri_gnn.config.const import METRICS

from hri_gnn.visualization import oracle

from hri_gnn.utils import paths

OUTPUT_DIR_PATH = paths.get_output_path()
DATASET_NAME = "raven-20000"
PERCENTAGE_TO_USE = 0.0025
NB_EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
LEARNING_DETAILS = True
METRICS_DETAILED = True




def print_and_save_validation_results(validation_results):
    """Prints the evaluation metrics in a formatted way."""
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
    
    # plot confusion matrix for val results
    oracle.plot_confusion_matrix_attributes(validation_results,splits=["val"], save_path=OUTPUT_DIR_PATH)

    # Saving validation results
    validation_results_path = os.path.join(OUTPUT_DIR_PATH, "validation_results.json")
    with open(validation_results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n  [OK] Validation results saved : {validation_results_path}")


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def run_training():
    """Configure and run the training process for the CNN encoder model on the RAVEN dataset."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # 1. Load datasets

    print(f"\nLoading dataset : {DATASET_NAME}")
    print(f"  Percentage to use : {PERCENTAGE_TO_USE*100}%")

    train_dataset = RavenDataset(
                        dataset=DATASET_NAME,
                        split='train',
                        percentage_limit=PERCENTAGE_TO_USE)
    val_dataset = RavenDataset(
                        dataset=DATASET_NAME,
                        split='val',
                        percentage_limit=PERCENTAGE_TO_USE)
    test_dataset = RavenDataset(
                        dataset=DATASET_NAME,
                        split='test',
                        percentage_limit=PERCENTAGE_TO_USE)

    # 2. DataLoaders

    print(f"\nBatch size : {BATCH_SIZE}")
    print(f"Train dataset loaded : {len(train_dataset)} problems")
    print(f"Val dataset loaded : {len(val_dataset)} problems")
    print(f"Test dataset loaded : {len(test_dataset)} problems")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=raven_flat_collate)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=raven_flat_collate)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=raven_flat_collate)

    # 3. Model
    model = RavenCnnEncoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModele : RavenCnnEncoder ({total_params:,} parametres)")
    
    # 4. Optimizer and Loss

    print(f"\n Optimizer : Adam")
    print(f" Learning rate : {LEARNING_RATE}")
    print(f" Loss : CompositeLoss with loss_detailed=True")
    print(f" Metrics : CompositeMetric with metrics_detailed=True")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CompositeLoss(loss_detailed=LEARNING_DETAILS)
    metrics  = CompositeMetric(metrics_detailed=METRICS_DETAILED, device=device)

    # 5. Training
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, metrics=metrics, device=device)
    print(f"epochs={NB_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

    checkpoint_dir = os.path.join(OUTPUT_DIR_PATH, "checkpoints")
    with TrainingTimer(model=model) as train_timer:
        history = trainer.fit(train_loader, test_loader, epochs=NB_EPOCHS, checkpoint_dir=checkpoint_dir, checkpoint_epochs=1)
    model_training_time = train_timer.elapsed

    # 6. Validation
    val_results = trainer.validate(val_loader)
    print_and_save_validation_results(val_results)

    # 7. Saving the training history
    history_path = os.path.join(OUTPUT_DIR_PATH, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  [OK] Training history saved : {history_path}")

    # 8. Learning curves
    for metric in METRICS:
        if metric == "confusion_matrix":
            oracle.plot_confusion_matrix_attributes(history, save_path=OUTPUT_DIR_PATH)
        curves_path = os.path.join(OUTPUT_DIR_PATH, f"{metric}_curves.png")
        oracle.plot_metric_curves(history, metric, save_path=curves_path)
    
    for attribute in ENTITY_ATTRIBUTES:
        attribute_curves = os.path.join(OUTPUT_DIR_PATH, f"{attribute}_curves.png")
        oracle.plot_attribute_metrics_curves(history, attribute, save_path=attribute_curves)
    

    # 9. Saving the final model
    models_dir_path = os.path.join(OUTPUT_DIR_PATH, "models")
    if not os.path.exists(models_dir_path):
        os.makedirs(models_dir_path)
    model_path = os.path.join(models_dir_path, "raven_cnn_encoder_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  [OK] Model saved : {model_path}")



    return history, val_results ,model_training_time


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    run_training()