from collections import defaultdict
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from hri_gnn.models import graph_builder_v2 as gb
from hri_gnn.metrics import gnn_accuracy

from hri_gnn.config.const import GRAPH_NODE_TYPES, GRAPH_EDGE_TYPES



class GnnTrainer:
    def __init__(self, gnn_model: nn.Module, cnn_model: nn.Module, optimizer: torch.optim.Optimizer, metrics= None, node_types= None, active_edge_types=None, include_columns=True, device="cpu", output_path= None):
        self.gnn_model = gnn_model
        self.cnn_model = cnn_model
        self.optimizer = optimizer
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
        # unpack the raw batch and build HeteroData dictionary and metadata 
        hetero_data_dict, meta_dict = gb.build_hetero_data_batch(raw_batch, self.cnn_model, include_columns=self.include_columns, device=self.device)
        
        # build PyG batch from HeteroData dictionary and metadata 
        pyg_batch, batched_layout_keys, keys_order = gb.build_pyg_batch(hetero_data_dict, meta_dict)
        
        # send pyg_batch to device
        # pyrefly: ignore [missing-attribute]
        pyg_batch = pyg_batch.to(self.device)

        # set the model to train mode
        self.gnn_model.train()

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass through the GNN model (message passing, scoring and loss)
        n_problems = len(keys_order)
        scores_per_problem, losses_per_problem, mean_loss = self.gnn_model(pyg_batch, batched_layout_keys, n_problems)
        
        # compute accuracy
        n_correct = gnn_accuracy(scores_per_problem, pyg_batch.y, self.gnn_model.candidate_ids)

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
            'n_correct': n_correct,
            'n_problems': n_problems,
        }
        # add number of nodes per node type 
        train_step_output.update({node_type : pyg_batch[node_type].x.shape[0] for node_type in self.node_types})
        # add number of edges per edge type
        train_step_output.update({edge_type : pyg_batch[edge_type].edge_index.shape[1] for edge_type in self.active_edge_types})

        return train_step_output
    
    def test_step(self, raw_batch):
        """
        Performs one test step on a single batch.
        """
        # unpack the raw batch and build HeteroData dictionary and metadata 
        hetero_data_dict, meta_dict = gb.build_hetero_data_batch(raw_batch, self.cnn_model, include_columns=self.include_columns, device=self.device)
        
        # build PyG batch from HeteroData dictionary and metadata 
        pyg_batch, batched_layout_keys, keys_order = gb.build_pyg_batch(hetero_data_dict, meta_dict)
        
        # send pyg_batch to device
        # pyrefly: ignore [missing-attribute]
        pyg_batch = pyg_batch.to(self.device)

        # set the model to train mode
        self.gnn_model.eval()

        # forward pass through the GNN model (message passing, scoring and loss)
        n_problems = len(keys_order)
        with torch.inference_mode():
            scores_per_problem, losses_per_problem, mean_loss = self.gnn_model(pyg_batch, batched_layout_keys, n_problems)
        
        # compute accuracy
        n_correct = gnn_accuracy(scores_per_problem, pyg_batch.y, self.gnn_model.candidate_ids)

        # prepare output dict with per-batch loss and metrics
        test_step_output = {
            'loss': mean_loss.item() * n_problems,
            'n_correct': n_correct,
            'n_problems': n_problems,
        }

        # add number of nodes per node type
        test_step_output.update({node_type : pyg_batch[node_type].x.shape[0] for node_type in self.node_types})

        # add number of edges per edge type
        test_step_output.update({edge_type : pyg_batch[edge_type].edge_index.shape[1] for edge_type in self.active_edge_types})

        return test_step_output

    def fit(self, train_dataloader, test_dataloader, epochs, max_grad_norm=1.0, checkpoint_name_prefix=None, check_frequency=5, save_final_version= False, verbose=True, train_only=False):
        
        history = defaultdict(list)
        # history[f'train_loss'] = []
        # history[f'test_loss'] = []

        for epoch in tqdm(range(epochs)):
            train_loss_sums = {'loss': 0.0}
            train_metrics_sums = {'accuracy': 0.0}
            train_metrics_sums.update({node_type: 0 for node_type in self.node_types})
            train_metrics_sums.update({edge_type: 0 for edge_type in self.active_edge_types})
            train_num_problems = 0
            test_loss_sums = {'loss': 0.0}
            test_metrics_sums = {'accuracy': 0.0}
            test_metrics_sums.update({node_type: 0 for node_type in self.node_types})
            test_metrics_sums.update({edge_type: 0 for edge_type in self.active_edge_types})
            test_num_problems = 0

            for train_batch_id, train_batch in enumerate(train_dataloader):
                train_step_output = self.train_step(train_batch,max_grad_norm)
                train_loss_sums['loss'] += train_step_output['loss']
                train_metrics_sums['accuracy'] += train_step_output['n_correct']
                train_num_problems += train_step_output['n_problems']
                for node_type in self.node_types:
                    train_metrics_sums[node_type] += train_step_output[node_type]
                for edge_type in self.active_edge_types:
                    # pyrefly: ignore [bad-index, unsupported-operation]
                    train_metrics_sums[edge_type] += train_step_output[edge_type]

            if not train_only:
                for test_batch_id, test_batch in enumerate(test_dataloader):
                    test_step_output = self.test_step(test_batch)
                    test_loss_sums['loss'] += test_step_output['loss']
                    test_metrics_sums['accuracy'] += test_step_output['n_correct']
                    test_num_problems += test_step_output['n_problems']
                    for node_type in self.node_types:
                        test_metrics_sums[node_type] += test_step_output[node_type]
                    for edge_type in self.active_edge_types:
                         # pyrefly: ignore [bad-index, unsupported-operation]
                        test_metrics_sums[edge_type] += test_step_output[edge_type]
                
            train_avg_loss = train_loss_sums['loss'] / train_num_problems
            train_accuracy = train_metrics_sums['accuracy'] / train_num_problems
            
            if not train_only:
                test_avg_loss = test_loss_sums['loss'] / test_num_problems
                test_accuracy = test_metrics_sums['accuracy'] / test_num_problems

            history[f'train_loss'].append(train_avg_loss)
            history[f'train_accuracy'].append(train_accuracy)
            if not train_only:
                history[f'test_accuracy'].append(test_accuracy)
                history[f'test_loss'].append(test_avg_loss)
            if epoch == 0:
                history[f'train_number_of_problems'].append(train_num_problems)
                history[f'train_number_of_batches'] = train_batch_id+1
                if not train_only:
                    history[f'test_number_of_problems'].append(test_num_problems)
                    history[f'test_number_of_batches'] = test_batch_id+1
                for node_type in self.node_types:
                    history[f'train_{node_type}'].append(train_metrics_sums[node_type])
                    if not train_only:
                        history[f'test_{node_type}'].append(test_metrics_sums[node_type])
                for edge_type in self.active_edge_types:
                 # pyrefly: ignore [bad-index, unsupported-operation]
                    history[f'train_{edge_type}'].append(train_metrics_sums[edge_type])
                    if not train_only:
                        # pyrefly: ignore [bad-index, unsupported-operation]
                        history[f'test_{edge_type}'].append(test_metrics_sums[edge_type])

            if verbose:
                if train_only:
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f} | Train Acc: {train_accuracy:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Loss: {test_avg_loss:.4f} | Test Acc: {test_accuracy:.4f}')

            # saving model weights
            if self.output_path and check_frequency != 0:
                prefix  = "gnn_model" if checkpoint_name_prefix is None else "gnn_model_"+checkpoint_name_prefix   
                if (epoch + 1) % check_frequency == 0:
                    checkpoint_path = self.output_path / "checkpoints" / f"{prefix}_checkpoint_epoch_{epoch+1}.pth"
                    torch.save(self.gnn_model.state_dict(), checkpoint_path)
                    print(f' > Saved checkpoint: {checkpoint_path}')
                    if save_final_version and (epoch + 1) == epochs:
                        final_model_path = self.output_path / "models/ "f"{prefix}_final.pth"
                        torch.save(self.gnn_model.state_dict(), final_model_path)
                        print(f' > Saved final model: {final_model_path}')

        # Printing the end of training
        if train_only:
            output_string = "\nTraining finished!....\n"
            output_string += f"- Train number of problems: {history['train_number_of_problems']}\n"
            output_string += f"- Train number of nodes per node type: {dict((node_type, history[f'train_{node_type}']) for node_type in self.node_types)}\n"
            output_string += f"- Train number of edges per edge type: {dict((edge_type, history[f'train_{edge_type}']) for edge_type in self.active_edge_types)}\n"
        else:
            output_string = "\nTraining finished!....\n"
            output_string += f"- Train (Test) number of problems: {history['train_number_of_problems']} ({history['test_number_of_problems']})\n"
            output_string += f"- Train (Test) number of nodes per node type: {dict((node_type, history[f'train_{node_type}']) for node_type in self.node_types)} ({dict((node_type, history[f'test_{node_type}']) for node_type in self.node_types)})\n"
            output_string += f"- Train (Test) number of edges per edge type: {dict((edge_type, history[f'train_{edge_type}']) for edge_type in self.active_edge_types)} ({dict((edge_type, history[f'test_{edge_type}']) for edge_type in self.active_edge_types)})\n"
            output_string += f"- Train (Test) number of batches: {history['train_number_of_batches']} ({history['test_number_of_batches']})"
        
        print(output_string)
        
        return history


        
    