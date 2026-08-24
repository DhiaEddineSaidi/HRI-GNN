import torch
import torch.nn as nn
from tqdm.auto import tqdm

from hri_gnn.metrics import CompositeMetric
from hri_gnn.config.const import METRICS, ENTITY_ATTRIBUTES


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
    device : torch.device
        The device (CPU or GPU).
    """

    def __init__(self, model: nn.Module, optimizer:torch.optim.Optimizer, criterion:nn.Module, metrics= METRICS, metrics_detials= False, device= None):
        self.model = model
        # self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.metrics_detials = metrics_detials
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"No device provided, Trainer will use: {self.device}")
        else:
            self.device = device
            print(f"Device provided, Trainer will use: {self.device}")

        # Initialize metrics
        self.composite_metrics = CompositeMetric(metric_names=self.metrics, details=self.metrics_detials, device=device)

    def train_step(self, X_train, y_train):
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
        metrics = self.composite_metrics.compute(logits, y_train)
        composite_loss = losses['composit']

        # backward pass
        composite_loss.backward()
        # update the weights
        self.optimizer.step()

        return {"losses": losses, "metrics": metrics}

    def test_step(self, X_test, y_test):
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
        # set the model to eval mode
        self.model.eval()
        # move the data to the device (device agnostic code)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        # compute the loss and metrics in inference mode
        with torch.inference_mode():
            logits = self.model(X_test)
            losses = self.criterion(logits, y_test)
            metrics = self.composite_metrics.compute(logits, y_test)

        return {
            "losses": losses, 
            "metrics": metrics
        }

    def fit(self, train_dataloader, test_dataloader, epochs, checkpoint_dir=None, verbose=True):
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
        history = {
            'train_loss_composit': [], 'train_loss_type': [], 'train_loss_size': [], 'train_loss_color': [], 'train_loss_angle': [],
            'test_loss_composit': [], 'test_loss_type': [], 'test_loss_size': [], 'test_loss_color': [], 'test_loss_angle': []
        }
        for metric in self.metrics:
            if self.metrics_detials:
                for attribute in {"overall"} | ENTITY_ATTRIBUTES.keys():
                    history[f"train_{metric}_{attribute}"] = []
                    history[f"test_{metric}_{attribute}"] = []
            else:
                history[f"train_{metric}"] = []
                history[f"test_{metric}"] = []


        

        for epoch in tqdm(range(epochs)):
            # --- Entraînement ---
            train_loss_sums = {'composit': 0.0, 'type': 0.0, 'size': 0.0, 'color': 0.0, 'angle':0.0}
            if self.metrics_detials:
                train_metrics_sums = {metric:{attribute:0.0 for attribute in {"overall"} | ENTITY_ATTRIBUTES.keys()} for metric in self.metrics}
            else:
                train_metrics_sums = {metric: 0.0 for metric in self.metrics}
            num_train_batches = 0

            for batch in train_dataloader:
                X_train, y_train = batch[0], batch[1]
                step_output = self.train_step(X_train, y_train)
                for key in train_loss_sums.keys():
                    train_loss_sums[key] += step_output["losses"][key]
                
                for metric in train_metrics_sums.keys():
                    if self.metrics_detials:
                        # pyrefly: ignore [not-iterable]
                        for attribute in train_metrics_sums[metric].keys():
                            train_metrics_sums[metric][attribute] += step_output["metrics"][metric][attribute]
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        train_metrics_sums[metric] += step_output["metrics"][metric]
                        
                num_train_batches += 1

            # Moyennes d'entraînement
            for key in train_loss_sums:
                avg = train_loss_sums[key] / max(num_train_batches, 1)
                history[f'train_loss_{key}'].append(avg)

            for metric in train_metrics_sums:
                if self.metrics_detials:
                    # pyrefly: ignore [not-iterable]
                    for attribute in train_metrics_sums[metric]:
                        # pyrefly: ignore [not-iterable]
                        avg = train_metrics_sums[metric][attribute] / max(num_train_batches, 1)
                        history[f'train_{metric}_{attribute}'].append(avg)
                else:
                    # pyrefly: ignore [unsupported-operation]
                    avg = train_metrics_sums[metric] / max(num_train_batches, 1)
                    history[f'train_{metric}'].append(avg)

            # --- Test ---
            test_sums = {'composit': 0.0, 'type': 0.0, 'size': 0.0, 'color': 0.0, 'angle':0.0}
            if self.metrics_detials:
                test_metrics_sums = {metric:{attribute:0.0 for attribute in {"overall"} | ENTITY_ATTRIBUTES.keys()} for metric in self.metrics}
            else:
                test_metrics_sums = {metric: 0.0 for metric in self.metrics}
            num_test_batches = 0

            for batch in test_dataloader:
                X_test, y_test = batch[0], batch[1]
                test_output = self.test_step(X_test, y_test)
                for key in test_sums:
                    test_sums[key] += test_output["losses"][key]
                
                for metric in test_metrics_sums.keys():
                    if self.metrics_detials:
                        # pyrefly: ignore [not-iterable]
                        for attribute in test_metrics_sums[metric].keys():
                            test_metrics_sums[metric][attribute] += test_output["metrics"][metric][attribute]
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        test_metrics_sums[metric] += test_output["metrics"][metric]
                        
                num_test_batches += 1

            # Moyennes de validation
            for key in test_sums:
                avg = test_sums[key] / max(num_test_batches, 1)
                history[f'test_loss_{key}'].append(avg)

            for metric in test_metrics_sums:
                if self.metrics_detials:
                    # pyrefly: ignore [not-iterable]
                    for attribute in test_metrics_sums[metric].keys():
                        # pyrefly: ignore [not-iterable]
                        avg = test_metrics_sums[metric][attribute] / max(num_test_batches, 1)
                        history[f'test_{metric}_{attribute}'].append(avg)
                else:
                    # pyrefly: ignore [unsupported-operation]
                    avg = test_metrics_sums[metric] / max(num_test_batches, 1)
                    history[f'test_{metric}'].append(avg)

            # Affichage
            string_to_print = 'Train (Test) losses: '
            for key in train_loss_sums:
                t = history[f'train_loss_{key}'][-1]
                v = history[f'test_loss_{key}'][-1]
                string_to_print += f"{key}: {t:.4f} ({v:.4f}) - "
            string_to_print +="\n Train (Test) Metrics :"
            if verbose:
                for metric in test_metrics_sums:
                    if self.metrics_detials:
                        metric_dict = test_metrics_sums[metric]
                        for attribute in metric_dict.keys():
                            t = history[f'train_{metric}_{attribute}'][-1]
                            v = history[f'test_{metric}_{attribute}'][-1]
                            string_to_print += f"{attribute}: {t:.4f} ({v:.4f}) - "
                    else:
                        t = history[f'train_{metric}'][-1]
                        v = history[f'test_{metric}'][-1]
                        string_to_print += f"{metric}: {t:.4f} ({v:.4f}) - "
            print(f"Epoque {epoch+1}/{epochs}: \n- {string_to_print}")

            # Checkpoint toutes les 5 époques
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), path)
                print(f"  > Checkpoint sauvegarde : {path}")

        return history

    def evaluate(self, val_dataloader):
        """
        evaluate the model on a val dataset and calculate detailed metrics.

        Args
        ----
        val_dataloader : DataLoader
            val dataset loader.

        Returns
        -------
        dict
            Métriques : 'loss_total', 'loss_presence', 'loss_attr', 'loss_recon',
            'presence_accuracy', 'type_accuracy', 'size_mae', 'color_mae', 'angle_mae',
            'recon_mse'
        """
        self.model.eval()

        # Accumulateurs pour les pertes
        loss_sums = {'total': 0.0, 'presence': 0.0, 'attr': 0.0, 'recon': 0.0}
        num_batches = 0

        # Accumulateurs pour les métriques
        presence_correct = 0
        presence_total = 0
        type_correct = 0
        type_total = 0
        size_ae_sum = 0.0
        color_ae_sum = 0.0
        angle_ae_sum = 0.0
        regression_count = 0
        recon_mse_sum = 0.0
        recon_count = 0

        with torch.no_grad():
            for batch in test_loader:
                images, targets = batch[0].to(self.device), batch[1].to(self.device)
                pred_z, pred_img = self.model(images)

                # Pertes
                losses = self.criterion(pred_z, pred_img, targets, images)
                for key in loss_sums:
                    loss_sums[key] += losses[key].item()
                num_batches += 1

                # --- Métriques de présence ---
                pred_presence = (torch.sigmoid(pred_z[:, :, 0]) > 0.5).float()
                target_presence = targets[:, :, 0]
                presence_correct += (pred_presence == target_presence).sum().item()
                presence_total += target_presence.numel()

                # --- Métriques sur les slots actifs ---
                mask = (target_presence == 1)
                if mask.any():
                    # Précision du type (argmax sur colonnes 2:8)
                    pred_types = pred_z[mask][:, 2:8].argmax(dim=1)
                    target_types = targets[mask][:, 2:8].argmax(dim=1)
                    type_correct += (pred_types == target_types).sum().item()
                    type_total += target_types.numel()

                    # MAE des attributs de régression
                    pred_attrs = torch.sigmoid(pred_z[mask])
                    size_ae_sum += (pred_attrs[:, 8] - targets[mask][:, 8]).abs().sum().item()
                    color_ae_sum += (pred_attrs[:, 9] - targets[mask][:, 9]).abs().sum().item()
                    angle_ae_sum += (pred_attrs[:, 12] - targets[mask][:, 12]).abs().sum().item()
                    regression_count += mask.sum().item()

                # Reconstruction MSE
                recon_mse_sum += F.mse_loss(pred_img, images, reduction='sum').item()
                recon_count += images.numel()

        metrics = {
            'loss_total': loss_sums['total'] / max(num_batches, 1),
            'loss_presence': loss_sums['presence'] / max(num_batches, 1),
            'loss_attr': loss_sums['attr'] / max(num_batches, 1),
            'loss_recon': loss_sums['recon'] / max(num_batches, 1),
            'presence_accuracy': presence_correct / max(presence_total, 1),
            'type_accuracy': type_correct / max(type_total, 1),
            'size_mae': size_ae_sum / max(regression_count, 1),
            'color_mae': color_ae_sum / max(regression_count, 1),
            'angle_mae': angle_ae_sum / max(regression_count, 1),
            'recon_mse': recon_mse_sum / max(recon_count, 1),
        }

        return metrics