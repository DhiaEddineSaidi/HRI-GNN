import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm

#from hri_gnn.metrics import CompositeMetric
from hri_gnn.config.const import ENTITY_ATTRIBUTES


class Trainer:
    """
    Trainer for the RavenCnnEncoder model.

    Paramètres
    ----------
    model : nn.Module
        The model to train (RavenEncoderDecoder).
    optimizer : torch.optim.Optimizer
        The optimizer (ex: Adam).
    criterion : nn.Module
        The loss function (CompositeLoss).

    """

    def __init__(self, model: nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, metrics, device= "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics
        # pyrefly: ignore [not-callable]
        self.loss_detailed = self.criterion.is_loss_detailed()
        self.metrics_detailed = self.metrics.is_metrics_detailed()

    def train_step(self, batch):
        """
        Execute a training step on a batch.

        Paramètres
        ----------
        X_train : Tensor (B, 1, 80, 80) of crops images
        y_train : Tensor (B, 4) ground truth of vector attributes

        Retourne
        --------
        dict of the losses and metrics
        """

        # unpack the batch
        X_train, y_train = batch[0], batch[1]

        # set the model to train mode
        self.model.train()

        # move the data to the device (device agnostic code)
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # zero the gradients
        self.optimizer.zero_grad()

        # forward pass
        logits = self.model(X_train)

        # compute the loss
        losses = self.criterion(logits, y_train)
        # compute metrics
        metrics = self.metrics.compute(logits, y_train)
        composite_loss = losses['loss']

        # backward pass
        composite_loss.backward()
        # update the weights
        self.optimizer.step()
        
        train_step_output = {}
        train_step_output.update(losses)
        train_step_output.update(metrics)

        return train_step_output

    def test_step(self,batch):
        """
        Execute an test step on a batch (without gradients).

        Args
        ----
        X_test : Tensor (B, 1, 160, 160)
        y_test : Tensor (B, 4)

        Returns
        -------
        dict of the losses and metrics
        """
        X_test, y_test = batch[0], batch[1]
        # set the model to eval mode
        self.model.eval()
        # move the data to the device (device agnostic code)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        # compute the loss and metrics in inference mode
        with torch.inference_mode():
            logits = self.model(X_test)
            losses = self.criterion(logits, y_test)
            metrics = self.metrics.compute(logits, y_test)

        test_step_output = {}
        test_step_output.update(losses)
        test_step_output.update(metrics)

        return test_step_output

    def fit(self, train_dataloader, test_dataloader, epochs, checkpoint_dir=None, checkpoint_epochs=5, verbose=True):
        """
        Complete training loop over several epochs.

        Args
        ----
        train_dataloader : DataLoader
            Training data loader.
        test_dataloader : DataLoader
            Validation data loader.
        epochs : int
            Number of epochs.
        checkpoint_dir : str, optional
            Directory to save checkpoints.

        Returns
        -------
        dict
            History of losses with the keys:
            'train_total', 'train_type', 'train_size', 'train_color', 'train_angle',
            'test_total', 'test_type', 'test_size', 'test_color', 'test_angle'
            Each value is a list of floats (one per epoch).
        """
        # initialize history
        history = {}
        history[f"train_loss"] = []
        history[f"test_loss"] = []

        if self.loss_detailed:
            for attribute in ENTITY_ATTRIBUTES.keys():
                history[f"train_loss_{attribute}"] = []
                history[f"test_loss_{attribute}"] = []

        for metric in self.metrics:
            history[f"train_{metric}"] = []
            history[f"test_{metric}"] = []
            if self.metrics_detailed:
                for attribute in ENTITY_ATTRIBUTES.keys():
                    history[f"train_{metric}_{attribute}"] = []
                    history[f"test_{metric}_{attribute}"] = []

        num_train_crops = 0
        num_test_crops = 0
        
        for epoch in tqdm(range(epochs)):

            # --- Training ---

            # initialize the training loss sums
            train_loss_sums = {'loss': 0.0}
            if self.loss_detailed:
                for attribute in ENTITY_ATTRIBUTES.keys():
                    train_loss_sums[f'loss_{attribute}'] = 0.0
            
            # initialize the training metrics sums
            train_metrics_sums = {metric:0.0 for metric in self.metrics}
            if self.metrics_detailed:
                 train_metrics_sums.update({f'{metric}_{attribute}':0.0 for metric in self.metrics for attribute in ENTITY_ATTRIBUTES.keys()})
                
            # start the trianing loop
            for train_batch_id, train_batch in enumerate(train_dataloader):
            
                # execute the training step
                step_output = self.train_step(train_batch)

                # update the loss sums
                for key in train_loss_sums.keys():
                    train_loss_sums[key] += step_output[key].item()
                
                # update the metrics sums
                for key in train_metrics_sums.keys():
                    train_metrics_sums[key] += step_output[key]
                        
                # keep track of the number of crops
                if epoch == 0:
                    num_train_crops += train_batch[0].size(0)
            
            if epoch == 0:
                # add num_of_crops and num_of_batches to history
                # pyrefly: ignore [unsupported-operation]
                history['train_number_of_crops']= num_train_crops
                # pyrefly: ignore [unsupported-operation]
                history['train_number_of_batches']= train_batch_id+1

            # training averages
            for key in train_loss_sums:
                train_epoch_avg_loss = train_loss_sums[key] / (train_batch_id+1)
                history[f'train_{key}'].append(train_epoch_avg_loss)

            for key in train_metrics_sums:
                train_epoch_avg_metric = train_metrics_sums[key] / (train_batch_id+1)
                history[f'train_{key}'].append(train_epoch_avg_metric)

            # ----------Test----------
            
            # initialize the test loss sums
            test_loss_sums = {'loss': 0.0}
            if self.loss_detailed:
                for attribute in ENTITY_ATTRIBUTES.keys():
                    test_loss_sums[f'loss_{attribute}'] = 0.0
            # initialize the test metrics sums
            test_metrics_sums = {metric:0.0 for metric in self.metrics}
            if self.metrics_detailed:
                 test_metrics_sums.update({f'{metric}_{attribute}':0.0 for metric in self.metrics for attribute in ENTITY_ATTRIBUTES.keys()})
                
            for test_batch_id, test_batch in enumerate(test_dataloader):
                test_output = self.test_step(test_batch)
                
                # update the test loss sums
                for key in test_loss_sums.keys():
                    test_loss_sums[key] += test_output[key].item()
                
                # update the test metrics sums
                for metric in test_metrics_sums.keys():
                    test_metrics_sums[metric] += test_output[metric]
                
                # keep track of the number of crops
                if epoch == 0:
                    num_test_crops += test_batch[0].size(0)
            
            if epoch == 0:
                # add num_of_crops and num_of_batches to history
                # pyrefly: ignore [unsupported-operation]
                history['test_number_of_crops'] =num_test_crops
                # pyrefly: ignore [unsupported-operation]
                history['test_number_of_batches'] =test_batch_id+1

            # test averages
            for key in test_loss_sums.keys():
                test_avg_loss = test_loss_sums[key] / (test_batch_id+1)
                history[f'test_{key}'].append(test_avg_loss)
            
            for metric in test_metrics_sums.keys():
                test_avg_metric = test_metrics_sums[metric] / (test_batch_id+1)
                history[f'test_{metric}'].append(test_avg_metric)

            # Printing
            string_to_print = "- Train (Test) losses: "
            for key in train_loss_sums:
                t = history[f'train_{key}'][-1]
                v = history[f'test_{key}'][-1]
                string_to_print += f"{key}: {t:.4f} ({v:.4f}) - "
            
            # Adding metrics to the string to print
            if verbose:
                string_to_print +="\n- Train (Test) Metrics :"
                for key in train_metrics_sums.keys():
                    t = history[f'train_{key}'][-1]
                    v = history[f'test_{key}'][-1]
                    string_to_print += f"{key}: {t:.4f} ({v:.4f}) - "
            
            print(f"Epoque {epoch+1}/{epochs}: \n{string_to_print}")

            # Saving a checkpoint every 'checkpoint_epochs'
            if checkpoint_dir and (epoch + 1) % checkpoint_epochs == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  > Checkpoint saved: {path}")

        # Printing the end of training
        output_string = "\nTraining finished!....\n"
        output_string += f"Train (Test) number of crops: {history['train_number_of_crops']} ({history['test_number_of_crops']})"
        output_string += f"Train (Test) number of batches: {history['train_number_of_batches']} ({history['test_number_of_batches']})"
        print(output_string)

        return history

    def validate(self, val_dataloader):
        """
        validate the model on a validation dataloader and calculate detailed metrics.

        Args    
        ----
        val_dataloader : DataLoader
            validation dataloader.

        Returns
        -------
        dict
            - 'loss': composite loss
            - 'type_loss': type loss
            - 'size_loss': size loss
            - 'color_loss': color loss
            - 'angle_loss': angle loss 
            - 'accuracy': overall accuracy
            - 'accuracy_type': type accuracy
            - 'accuracy_size': size accuracy
            - 'accuracy_color': color accuracy
            - 'accuracy_angle': angle accuracy
            - 'accuracy': overall accuracy
            - 'precision': overall precision 
            - 'precision_type': type precision
            - 'precision_size': size precision
            - 'precision_color': color precision
            - 'precision_angle': angle precision
            - 'precision': overall precision
            - 'recall': overall recall 
            - 'recall_type': type recall
            - 'recall_size': size recall
            - 'recall_color': color recall
            - 'recall_angle': angle recall
            - 'recall': overall recall
            - 'f1_score': overall f1-score
            - 'f1_type': type f1-score
            - 'f1_size': size f1-score
            - 'f1_color': color f1-score
            - 'f1_angle': angle f1-score
            - 'f1': overall f1-score
            - 'num_of_crops': number of crops
            - 'num_of_batches': number of batches
        y_true: torch.Tensor
            true values
        y_logits: dict
            predicted logits
        """
        self.model.eval()

        # accumulators for loss
        loss_sums = {'loss': 0.0}
        if self.loss_detailed:
            for attribute in ENTITY_ATTRIBUTES.keys():
                loss_sums[f'loss_{attribute}'] = 0.0
        # accumulators for metrics
        metrics_sums = {metric:0.0 for metric in self.metrics}
        if self.metrics_detailed:
            metrics_sums.update({f'{metric}_{attribute}':0.0 for metric in self.metrics for attribute in ENTITY_ATTRIBUTES.keys()})
        num_batches = 0
        num_crops = 0

        batch_y_logits = {"pred_type_logits":[], "pred_size_logits":[], "pred_color_logits":[], "pred_angle_logits":[]}
        y_logits = {}
        y_true = []
        self.model.eval()
        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validating the model..."):
                X_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(X_val)

                # losses
                losses = self.criterion(logits,y_val)
                for key in loss_sums:
                    loss_sums[key] += losses[key].item()
                
                # metrics
                metrics = self.metrics.compute(logits, y_val)
                for key in metrics_sums:
                    metrics_sums[key] += metrics[key]

                # store logits and true values
                for key in batch_y_logits.keys():
                    batch_y_logits[key].append(logits[key])
                y_true.append(y_val)

                num_batches += 1
                num_crops += X_val.size(0)
        
        # Printing the validation results
        output_string = "\nValidation finished!....\n"
        output_string += f"Number of crops: {num_crops} | Number of batches: {num_batches}\n"
        print(output_string)

        results = {}
        # add losses to results
        for key in loss_sums:
            results[f'val_{key}'] = loss_sums[key] / max(num_batches, 1)

        # add metrics to results
        for metric in metrics_sums:
            results[f'val_{metric}'] = metrics_sums[metric] / max(num_batches, 1)
        
        results['val_num_of_crops'] = num_crops
        results['val_num_of_batches'] = num_batches

        # add logits and true values to results
        for key in batch_y_logits.keys():
            y_logits[key] = torch.cat(batch_y_logits[key], dim=0)

        return results , y_logits, torch.cat(y_true, dim=0)