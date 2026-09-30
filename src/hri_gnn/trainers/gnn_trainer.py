import json
from collections import defaultdict
import torch
import torch.nn as nn

# pyrefly: ignore [untyped-import]
from tqdm.auto import tqdm

from hri_gnn.models import graph_builder as gb
from hri_gnn.metrics import GnnMetrics

from hri_gnn.config.const import GRAPH_NODE_TYPES, GRAPH_EDGE_TYPES



class GnnTrainer:
    def __init__(self, gnn_model: nn.Module, cnn_model: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler=None, metrics=None, node_types=None, active_edge_types=None, 
                 include_columns=True, device="cpu", output_path=None):
        self.gnn_model = gnn_model
        self.cnn_model = cnn_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.device = device
        self.output_path = output_path

        # TODO: validate the combination of node_types and active_edge_types
        self.node_types = [node_type for node_type in GRAPH_NODE_TYPES.keys()] if node_types is None else node_types
        self.active_edge_types = [k for k in GRAPH_EDGE_TYPES.keys() if k in active_edge_types] if active_edge_types is not None else list(GRAPH_EDGE_TYPES.keys())
        self.include_columns = include_columns

        # TODO: make the graph builder more flexible, tell it which node types to build and which edge types to include

    def train_step(self, raw_batch, max_grad_norm= None):
        """
        Performs one training step on a single batch.

        """
        if isinstance(raw_batch, tuple) and len(raw_batch) == 4 and hasattr(raw_batch[0], 'num_graphs'):
            # Precomputed PyG batch
            pyg_batch, batched_layout_keys, all_configs, all_problem_ids = raw_batch
            n_problems = pyg_batch.num_graphs // len(self.gnn_model.candidate_ids)
            pyg_batch = pyg_batch.to(self.device)
        else:
            # unpack the raw batch and build HeteroData dictionary and metadata 
            hetero_data_dict, meta_dict = gb.build_hetero_data_batch(raw_batch, self.cnn_model, include_columns=self.include_columns, device=self.device)
            
            # build PyG batch from HeteroData dictionary and metadata 
            pyg_batch, batched_layout_keys, keys_order = gb.build_pyg_batch(hetero_data_dict, meta_dict)
            n_problems = len(keys_order) // len(self.gnn_model.candidate_ids)
            
            # send pyg_batch to device
            # pyrefly: ignore [missing-attribute]
            pyg_batch = pyg_batch.to(self.device)

        # set the model to train mode
        self.gnn_model.train()

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass through the GNN model (message passing, scoring and loss)
        scores_per_problem, losses_per_problem, mean_loss = self.gnn_model(pyg_batch, batched_layout_keys, n_problems)

        # backward pass
        mean_loss.backward()

        # gradient clipping to prevent exploding gradients
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), max_grad_norm)

        # update the weights
        self.optimizer.step()

        # prepare output dict with per-batch loss and metrics
        train_step_output = {
            'loss': mean_loss.item() * n_problems,
            'n_problems': n_problems,
            'scores_per_problem': scores_per_problem,
            'y': pyg_batch.y,
        }
        # add number of nodes per node type 
        train_step_output.update({node_type : pyg_batch[node_type].x.shape[0] for node_type in self.node_types})
        # add number of edges per edge type
        # pyrefly: ignore [no-matching-overload]
        train_step_output.update({edge_type : pyg_batch[edge_type].edge_index.shape[1] for edge_type in self.active_edge_types})

        return train_step_output
    
    def test_step(self, raw_batch):
        """
        Performs one test step on a single batch.
        """
        if isinstance(raw_batch, tuple) and len(raw_batch) == 4 and hasattr(raw_batch[0], 'num_graphs'):
            # Precomputed PyG batch
            pyg_batch, batched_layout_keys, all_configs, all_problem_ids = raw_batch
            n_problems = pyg_batch.num_graphs // len(self.gnn_model.candidate_ids)
            pyg_batch = pyg_batch.to(self.device)
        else:
            # unpack the raw batch and build HeteroData dictionary and metadata 
            hetero_data_dict, meta_dict = gb.build_hetero_data_batch(raw_batch, self.cnn_model, include_columns=self.include_columns, device=self.device)
            
            # build PyG batch from HeteroData dictionary and metadata 
            pyg_batch, batched_layout_keys, keys_order = gb.build_pyg_batch(hetero_data_dict, meta_dict)
            n_problems = len(keys_order) // len(self.gnn_model.candidate_ids)
            
            # send pyg_batch to device
            # pyrefly: ignore [missing-attribute]
            pyg_batch = pyg_batch.to(self.device)

        # set the model to train mode
        self.gnn_model.eval()

        # forward pass through the GNN model (message passing, scoring and loss)
        with torch.inference_mode():
            scores_per_problem, losses_per_problem, mean_loss = self.gnn_model(pyg_batch, batched_layout_keys, n_problems)

        # prepare output dict with per-batch loss and metrics
        test_step_output = {
            'loss': mean_loss.item() * n_problems,
            'n_problems': n_problems,
            'scores_per_problem': scores_per_problem,
            'y': pyg_batch.y,
        }

        # add number of nodes per node type
        test_step_output.update({node_type : pyg_batch[node_type].x.shape[0] for node_type in self.node_types})

        # add number of edges per edge type
        # pyrefly: ignore [no-matching-overload]
        test_step_output.update({edge_type : pyg_batch[edge_type].edge_index.shape[1] for edge_type in self.active_edge_types})

        return test_step_output

    def fit(self, train_dataloader, test_dataloader, epochs, max_grad_norm=1.0, checkpoint_name_prefix=None, check_frequency=5, save_final_version=False, verbose=True, history_save_path=None):
        
        history = defaultdict(list)

        best_test_acc = -1.0
        best_epoch = -1
        train_nb_batches = 0
        test_nb_batches = 0

        train_gnn_metrics = GnnMetrics(num_candidates=len(self.gnn_model.candidate_ids))
        test_gnn_metrics  = GnnMetrics(num_candidates=len(self.gnn_model.candidate_ids))

        for epoch in tqdm(range(epochs)):
            # train initializations
            train_loss_sum = 0
            train_metrics_sums = {}
            train_metrics_sums.update({node_type: 0 for node_type in self.node_types})
            # pyrefly: ignore [no-matching-overload]
            train_metrics_sums.update({edge_type: 0 for edge_type in self.active_edge_types})
            train_num_problems = 0
            
            # test initializations
            test_loss_sum = 0
            test_metrics_sums = {}
            test_metrics_sums.update({node_type: 0 for node_type in self.node_types})
            # pyrefly: ignore [no-matching-overload]
            test_metrics_sums.update({edge_type: 0 for edge_type in self.active_edge_types})
            test_num_problems = 0

            train_gnn_metrics.reset()
            for train_batch in train_dataloader:
                train_step_output = self.train_step(train_batch, max_grad_norm)
                train_loss_sum += train_step_output['loss']
                train_num_problems += train_step_output['n_problems']
                if epoch == 0:
                    train_nb_batches += 1
                
                train_gnn_metrics.update(
                    train_step_output['scores_per_problem'],
                    train_step_output['y'],
                    self.gnn_model.candidate_ids
                )
                for node_type in self.node_types:
                    train_metrics_sums[node_type] += train_step_output[node_type]
                for edge_type in self.active_edge_types:
                    # pyrefly: ignore [bad-index, unsupported-operation]
                    train_metrics_sums[edge_type] += train_step_output[edge_type]

            #if not train_only:
            test_gnn_metrics.reset()
            for test_batch in test_dataloader:
                test_step_output = self.test_step(test_batch)
                test_loss_sum += test_step_output['loss']
                test_num_problems += test_step_output['n_problems']
                if epoch == 0:
                    test_nb_batches += 1
                test_gnn_metrics.update(
                    test_step_output['scores_per_problem'],
                    test_step_output['y'],
                    self.gnn_model.candidate_ids
                )
                for node_type in self.node_types:
                    test_metrics_sums[node_type] += test_step_output[node_type]
                for edge_type in self.active_edge_types:
                     # pyrefly: ignore [bad-index, unsupported-operation]
                    test_metrics_sums[edge_type] += test_step_output[edge_type]
                
            train_avg_loss = train_loss_sum / train_num_problems
            test_avg_loss = test_loss_sum / test_num_problems
            
            # Compute epoch-level classification metrics
            train_epoch_metrics = train_gnn_metrics.compute()
            test_epoch_metrics  = test_gnn_metrics.compute()
            
            # Step the scheduler on test loss (only if scheduler is set)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_avg_loss)
                else:
                    self.scheduler.step()

            # Best-model checkpointing based on test accuracy
            if test_epoch_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_epoch_metrics['accuracy']
                best_epoch = epoch + 1
                if self.output_path:
                    prefix = "gnn_model" if checkpoint_name_prefix is None else "gnn_model_" + checkpoint_name_prefix
                    best_model_path = self.output_path / "models" / f"{prefix}_best.pth"
                    torch.save(self.gnn_model.state_dict(), best_model_path)

            history[f'train_loss'].append(train_avg_loss)
            history[f'test_loss'].append(test_avg_loss)

            # store scalar metrics in history (skip confusion matrix)
            for k, v in train_epoch_metrics.items():
                if k != 'confusion_matrix':
                    history[f'train_{k}'].append(v)
            for k, v in test_epoch_metrics.items():
                if k != 'confusion_matrix':
                    history[f'test_{k}'].append(v)
            # store only the last epoch's confusion matrix (tensors are large)
            history['train_confusion_matrix'] = train_epoch_metrics['confusion_matrix'].tolist()
            history['test_confusion_matrix']  = test_epoch_metrics['confusion_matrix'].tolist()
            if epoch == 0:
                history[f'train_number_of_problems'].append(train_num_problems)
                # pyrefly: ignore [unsupported-operation]
                history[f'train_number_of_batches'] = train_nb_batches
                # if not train_only:
                history[f'test_number_of_problems'].append(test_num_problems)
                # pyrefly: ignore [unsupported-operation]
                history[f'test_number_of_batches'] = test_nb_batches
                for node_type in self.node_types:
                    history[f'train_{node_type}'].append(train_metrics_sums[node_type])
                    history[f'test_{node_type}'].append(test_metrics_sums[node_type])
                for edge_type in self.active_edge_types:
                 # pyrefly: ignore [bad-index, unsupported-operation]
                    history[f'train_{edge_type}'].append(train_metrics_sums[edge_type])
                    # pyrefly: ignore [bad-index, unsupported-operation]
                    history[f'test_{edge_type}'].append(test_metrics_sums[edge_type])

            if verbose:
                lr_info = f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}" if self.scheduler is not None else ""
                best_marker = " ★" if (test_epoch_metrics['accuracy'] == best_test_acc) else ""
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f} | Train Acc: {train_epoch_metrics['accuracy']:.4f} | '
                      f'Test Loss: {test_avg_loss:.4f} | Test Acc: {test_epoch_metrics['accuracy']:.4f} | '
                      f'Test P: {test_epoch_metrics["precision"]:.4f} | Test R: {test_epoch_metrics["recall"]:.4f} | '
                      f'Test F1: {test_epoch_metrics["f1"]:.4f}{lr_info}{best_marker}')

            # periodic checkpoint saving
            if self.output_path and check_frequency != 0:
                prefix  = "gnn_model" if checkpoint_name_prefix is None else "gnn_model_"+checkpoint_name_prefix   
                if (epoch + 1) % check_frequency == 0:
                    checkpoint_path = self.output_path / "checkpoints" /"gnn"/ f"{prefix}_checkpoint_epoch_{epoch+1}.pth"
                    torch.save(self.gnn_model.state_dict(), checkpoint_path)
                    print(f' > Saved checkpoint: {checkpoint_path}')
                    # Save history alongside checkpoint so crash recovery is possible
                    if history_save_path is not None:
                        with open(history_save_path, 'w') as _hf:
                            json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in history.items()}, _hf, indent=2)
                        print(f' > Saved history  : {history_save_path}')
                    if save_final_version and (epoch + 1) == epochs:
                        final_model_path = self.output_path / "models" / f"{prefix}_final.pth"
                        torch.save(self.gnn_model.state_dict(), final_model_path)
                        print(f' > Saved final model: {final_model_path}')

        output_string = "\nTraining finished!....\n"
        output_string += f"- Train (Test) number of problems: {history['train_number_of_problems']} ({history['test_number_of_problems']})\n"
        output_string += f"- Train (Test) number of nodes per node type: {dict((node_type, history[f'train_{node_type}']) for node_type in self.node_types)} ({dict((node_type, history[f'test_{node_type}']) for node_type in self.node_types)})\n"
        output_string += f"- Train (Test) number of edges per edge type: {dict((edge_type, history[f'train_{edge_type}']) for edge_type in self.active_edge_types)} ({dict((edge_type, history[f'test_{edge_type}']) for edge_type in self.active_edge_types)})\n"
        output_string += f"- Train (Test) number of batches: {history['train_number_of_batches']} ({history['test_number_of_batches']})"
        
        print(output_string)
        
        return history

    def validate(self, val_dataloader):
        """
        Validate the model on a validation dataloader and calculate detailed metrics.

        Args
        ----
        val_dataloader : DataLoader
            Validation dataloader (can be test or val split).

        Returns
        -------
        dict
            - 'val_loss': average loss per problem
            - 'val_accuracy': accuracy
            - 'val_precision': macro-averaged precision
            - 'val_recall': macro-averaged recall
            - 'val_f1': macro-averaged F1
            - 'val_confusion_matrix': (8x8) confusion matrix as list
            - 'val_num_problems': total number of problems
            - 'val_num_batches': total number of batches
        """
        self.gnn_model.eval()
        
        gnn_metrics = GnnMetrics(num_candidates=len(self.gnn_model.candidate_ids))
        
        total_problems = 0
        num_batches = 0

        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validating..."):
                if isinstance(batch, tuple) and len(batch) == 4 and hasattr(batch[0], 'num_graphs'):
                    pyg_batch, batched_layout_keys, all_configs, all_problem_ids = batch
                    n_problems = pyg_batch.num_graphs // len(self.gnn_model.candidate_ids)
                    pyg_batch = pyg_batch.to(self.device)
                else:
                    hetero_data_dict, meta_dict = gb.build_hetero_data_batch(
                        batch, self.cnn_model, include_columns=self.include_columns, device=self.device
                    )
                    pyg_batch, batched_layout_keys, keys_order = gb.build_pyg_batch(hetero_data_dict, meta_dict)
                    n_problems = len(keys_order) // len(self.gnn_model.candidate_ids)
                    # pyrefly: ignore [missing-attribute]
                    pyg_batch = pyg_batch.to(self.device)

                scores_per_problem, losses_per_problem, mean_loss = self.gnn_model(
                    pyg_batch, batched_layout_keys, n_problems
                )

                gnn_metrics.update(scores_per_problem, pyg_batch.y, self.gnn_model.candidate_ids)

                total_problems += n_problems
                num_batches += 1

        # Compute epoch-level metrics
        metrics_results = gnn_metrics.compute()

        results = {
            'val_accuracy': metrics_results['accuracy'],
            'val_precision': metrics_results['precision'],
            'val_recall': metrics_results['recall'],
            'val_f1': metrics_results['f1'],
            'val_confusion_matrix': metrics_results['confusion_matrix'].tolist(),
            'val_num_problems': total_problems,
            'val_num_batches': num_batches,
        }

        # Print validation results
        output_string = "\nValidation finished!....\n"
        output_string += f"Number of problems: {total_problems} | Number of batches: {num_batches}\n"
        output_string += f"Accuracy: {results['val_accuracy']:.4f}\n"
        output_string += f"Precision: {results['val_precision']:.4f} | Recall: {results['val_recall']:.4f} | F1: {results['val_f1']:.4f}\n"
        print(output_string)

        return results
