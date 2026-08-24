import time
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score

from hri_gnn.config.const import ENTITY_ATTRIBUTES, METRICS


class CompositeMetric:
    def __init__(self, metric_names: list[str]=METRICS, details=False ,device=None):
        super().__init__()
        self.metric_names = metric_names
        self.details = details
        if device:
          self.device = device
        else:
          self.device = "cuda" if torch.cuda.is_available() else "cpu"
          print(f"No device was specified, CompositeMetric is setting device to: {self.device}")

        # Check for supported metrics
        unsupprted_metrics = set(self.metric_names) - set(METRICS)
        if unsupprted_metrics:
          raise ValueError(f"{unsupprted_metrics} are not supported. Supported metrics: {METRICS}")

        for metric in set(metric_names):
          match metric:
            case 'accuracy':
              self._create_accuracy_functions()
            case 'precision':
              self._create_precision_functions()
            case 'recall':
              self._create_recall_functions()
            case 'f1':
              self._create_f1_functions()
            case _:
              raise ValueError(f"Unknown metric: {metric}")
    
    def _convert_logits_to_pred(self, pred_logits):
      pred_type = pred_logits['pred_type_logits'].argmax(dim=1)
      pred_size = pred_logits['pred_size_logits'].argmax(dim=1)
      pred_color = pred_logits['pred_color_logits'].argmax(dim=1)
      pred_angle = pred_logits['pred_angle_logits'].argmax(dim=1)
      return {'type': pred_type, 'size': pred_size, 'color': pred_color, 'angle': pred_angle}
    
    def _create_targets_dict(self, y_true):
      return {'type': y_true[:,0], 'size': y_true[:,1], 'color': y_true[:,2], 'angle': y_true[:,3]}

    def _create_accuracy_functions(self):
      self._accuracy_fn_dict = {}
      for attribute in ENTITY_ATTRIBUTES.keys():
        self._accuracy_fn_dict[attribute] = MulticlassAccuracy(num_classes= len(ENTITY_ATTRIBUTES[attribute])).to(self.device)

    def _compute_accuracy(self, predictions, targets):
      """
      Compute the accuracy for each attribute.

      Returns
      -------
      dict: A dictionary with the accuracy for each attribute
      """
      accuracy_dict = {}
      for attribute, accuracy_fn in self._accuracy_fn_dict.items():
        accuracy_dict[attribute] = accuracy_fn(predictions[attribute], targets[attribute])
      accuracy = torch.mean(torch.tensor(list(accuracy_dict.values())))
      accuracy_dict["overall"] = accuracy
      return accuracy_dict

    def _create_precision_functions(self):
      self._precision_fn_dict = {}
      for attribute in ENTITY_ATTRIBUTES.keys():
        self._precision_fn_dict[attribute] = MulticlassPrecision(num_classes= len(ENTITY_ATTRIBUTES[attribute])).to(self.device)

    def _compute_precision(self, predictions, targets):
      """
      Compute the precision for each attribute.


      Returns
      -------
      dict: A dictionary with the precision for each attribute
      """
      precision_dict = {}
      for attribute, precision_fn in self._precision_fn_dict.items():
        precision_dict[attribute] = precision_fn(predictions[attribute], targets[attribute])
      precision = torch.mean(torch.tensor(list(precision_dict.values())))
      precision_dict["overall"] = precision
      return precision_dict

    def _create_recall_functions(self):
      self._recall_fn_dict = {}
      for attribute in ENTITY_ATTRIBUTES.keys():
        self._recall_fn_dict[attribute] = MulticlassRecall(num_classes= len(ENTITY_ATTRIBUTES[attribute])).to(self.device)

    def _compute_recall(self, predictions, targets):
      """
      Compute the recall for each attribute.

      Returns
      -------
      dict: A dictionary with the recall for each attribute
      """
      recall_dict = {}
      for attribute, recall_fn in self._recall_fn_dict.items():
        recall_dict[attribute] = recall_fn(predictions[attribute], targets[attribute])
      recall = torch.mean(torch.tensor(list(recall_dict.values())))
      recall_dict["overall"] = recall
      return recall_dict

    def _create_f1_functions(self):
      self._f1_fn_dict = {}
      for attribute in ENTITY_ATTRIBUTES.keys():
        self._f1_fn_dict[attribute] = MulticlassF1Score(num_classes= len(ENTITY_ATTRIBUTES[attribute])).to(self.device)

    def _compute_f1(self, predictions, targets):
      """
      Compute the f1 score for each attribute.

      Returns
      -------
      dict: A dictionary with the f1 score for each attribute
      """
      f1_dict = {}
      for attribute, f1_fn in self._f1_fn_dict.items():
        f1_dict[attribute] = f1_fn(predictions[attribute], targets[attribute])
      f1 = torch.mean(torch.tensor(list(f1_dict.values())))
      f1_dict["overall"] = f1
      return f1_dict

    def compute(self, pred_logits, y_true):
      
      predictions = self._convert_logits_to_pred(pred_logits)
      targets = self._create_targets_dict(y_true)

      results = {}
      for metric in self.metric_names:
        match metric:
          case 'accuracy':
            results[metric] = self._compute_accuracy(predictions,targets)
          case 'precision':
            results[metric] = self._compute_precision(predictions,targets)
          case 'recall':
            results[metric] = self._compute_recall(predictions,targets)
          case 'f1':
            results[metric] = self._compute_f1(predictions,targets)
      if self.details:
        return results
      else:
        return {metric: results[metric]["overall"] for metric in self.metric_names}


class TrainingTimer:
    """
    Timer context manager for training.
    
    Args
    ----
    model: nn.Module, the model to train
    """
    def __init__(self, model:nn.Module):
      self.model = model
      self.device = next(model.parameters()).device

    def __enter__(self):
      print(f"start training model of type: '{self.model.__class__.__name__}' on device: {self.device} ....")
      self.start = time.perf_counter()
      return self  # This object is bound to the 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
      self.end = time.perf_counter()
      self.elapsed = self.end - self.start
      print(f"training model of type '{self.model.__class__.__name__}' on device: {self.device}, took: {self.elapsed:.4f} seconds")
      return False

