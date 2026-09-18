import json

import torch 
from torch.utils.data import DataLoader

from hri_gnn.datasets.precomputed import PrecomputedRavenDataset, precomputed_collate
from hri_gnn.models.cnn_encoder import RavenCnnEncoder
from hri_gnn.models.gnn_encoder import RavenGnnModel 
from hri_gnn.trainers import GnnTrainer
from hri_gnn.metrics import GnnMetrics
from hri_gnn.metrics import TrainingTimer
from hri_gnn.utils import paths
from hri_gnn.config.const import GRAPH_NODE_TYPES




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
    print(f" Number: \n  - Crops   : {validation_results['val_num_problems']} \n  - Batches : {validation_results['val_num_batches']}")
    print("=" * 60)
    print(f"  Loss   : {validation_results['val_loss']:.4f}")
    print(f"  Accuracy : {validation_results['val_accuracy']*100:.2f}%")
    print(f"  Precision           : {validation_results['val_precision']:.4f}")
    print(f"  Recall           : {validation_results['val_recall']:.4f}")
    print(f"  F1 score          : {validation_results['val_f1']:.4f}")
    
    # Saving validation results
    if save_path:
        validation_results_path = save_path/"validation_results.json"
        with open(validation_results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\n[OK] Validation results saved : {validation_results_path}")

# =============================================================================
# Training Process
# =============================================================================


def run_training(dataset_name = 'raven-100', 
                 percentage_to_use=0.1, 
                 batch_size =256, 
                 nb_epochs= 150, 
                 check_frequency =10,                  
                 hidden_dim =64,
                 num_layers =3,
                 dropout=0.1,
                 learning_rate =5e-4,
                 weight_decay= 1e-4,
                 mode='min',
                 factor = 0.5,
                 patience=10,
                 min_lr=1e-5,
                 ):

    output_path = paths.get_output_path()
    paths.output_paths(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f" GPU : {torch.cuda.get_device_name(0)}")

    print(f"\nLoading dataset : {dataset_name}")
    print(f"  Percentage to use : {percentage_to_use*100}%")

    train_dataset = PrecomputedRavenDataset(dataset=dataset_name,
                                          split='train',
                                          percentage_limit=percentage_to_use,
                                          configs=None)

    val_dataset = PrecomputedRavenDataset(dataset=dataset_name,
                                          split='val',
                                          percentage_limit=percentage_to_use,
                                          configs=None)

    test_dataset = PrecomputedRavenDataset(dataset=dataset_name,
                                          split='test',
                                          percentage_limit=percentage_to_use,
                                          configs=None)

    print(f"\nBatch size : {batch_size}")
    print(f"Train dataset loaded : {len(train_dataset)} problems")
    print(f"Val dataset loaded : {len(val_dataset)} problems")
    print(f"Test dataset loaded : {len(test_dataset)} problems")

    train_dataloader = DataLoader(
                          dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle= True,
                          collate_fn=precomputed_collate,
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
    val_dataloader = DataLoader(
                          dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle= False,
                          collate_fn=precomputed_collate,
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
    test_dataloader = DataLoader(
                          dataset=val_dataset,
                          batch_size=batch_size,
                          shuffle= False,
                          collate_fn=precomputed_collate,
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
    
    # load the pre-trained CNN encoder model 
    raven_encoder = RavenCnnEncoder()
    raven_encoder.load_state_dict(torch.load(f=output_path/'models'/'raven_cnn_encoder_final.pth'))
    
    # GNN model 
    gnn_model = RavenGnnModel(GRAPH_NODE_TYPES, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to("cuda")
    # define the optimizer 
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # define the scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
    )
    
    # instanciate the GNN trainer 
    gnn_trainer = GnnTrainer(gnn_model, raven_encoder, optimizer, scheduler, 
                             metrics=None, node_types= None, active_edge_types= None, 
                             include_columns=True, device=device, output_path= output_path)

    # train the GNN model
    with TrainingTimer(model=gnn_model) as train_timer:
        history = gnn_trainer.fit(train_dataloader, test_dataloader, 
                                       epochs=nb_epochs, 
                                       max_grad_norm=1.0, 
                                       checkpoint_name_prefix = None,
                                       check_frequency=check_frequency,
                                       save_final_version= True,
                                       verbose=True)
    model_training_time = train_timer.elapsed

    # 6. Validation
    validation_results = gnn_trainer.validate(val_dataloader)
    validation_result_save_path = output_path/"history"/"gnn"
    print_validation_results(validation_results,save_path=validation_result_save_path)
    
    # 7. Saving the training history
    history_path = output_path/"history"/"gnn"/"training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved : {history_path}")

    return history, validation_results ,model_training_time