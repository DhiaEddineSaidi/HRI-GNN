import torch 
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):

    def __init__(self, criterion=nn.CrossEntropyLoss(), loss_detailed = True):
        """
        Initialize the composite loss.
        
        Args
        ----
        criterion: nn.Module
        """
        super().__init__()
        self.criterion = criterion
        self.loss_detailed = loss_detailed

    def is_loss_detailed(self):
      return self.loss_detailed

    def forward(self, pred_logits, y):
        """
        Compute the composite loss for the RavenCnnEncoder model, which is a multi task learning model, the loss is composed of the following:
        - type loss
        - size loss
        - color loss
        - angle loss

        Args
        -------
        pred_logits: tuple of torch.Tensor, the predictions for each task
        y: torch.Tensor, the targets for each task

        Returns
        -------
        dict with the following keys:
        - composite: torch.Tensor, the composite loss
        - type: torch.Tensor, the loss for the type task
        - size: torch.Tensor, the loss for the size task
        - color: torch.Tensor, the loss for the color task
        - angle: torch.Tensor, the loss for the angle task
        
        """
    
        # unpack the predictions logits
        #pred_type_logits , pred_size_logits, pred_color_logits, pred_angle_logits = pred_logits
        
        # convert y targets to long
        y_as_long = y.long()

        # extract the target for each task
        y_type = y_as_long[:,0]
        y_size = y_as_long[:,1]
        y_color = y_as_long[:,2]
        y_angle = y_as_long[:,3]

        # calculate the loss for each task
        loss_type = self.criterion(pred_logits['pred_type_logits'], y_type)
        loss_size = self.criterion(pred_logits['pred_size_logits'], y_size)
        loss_color = self.criterion(pred_logits['pred_color_logits'], y_color)
        loss_angle = self.criterion(pred_logits['pred_angle_logits'], y_angle)

        # calculate the composite loss (weigh equally for each task)
        composite_loss = loss_type + loss_size + loss_color + loss_angle

        if self.loss_detailed:
          return {
            "loss":composite_loss, 
            "loss_type":loss_type, 
            "loss_size":loss_size, 
            "loss_color":loss_color, 
            "loss_angle":loss_angle
          }
        else:
          return {
            "loss":composite_loss
          }


def compute_candidate_loss(scores: torch.Tensor, y: torch.Tensor, candidate_ids)-> torch.Tensor:
    """
    Computes the cross-entropy loss for the candidate prediction.
    
    Args:
      scores: torch.Tensor of shape (number of candidates, 1)
      y: torch.Tensor of shape (1,) containing the true candidate id
      candidate_ids: list of candidate ids
      
    Returns:
      torch.Tensor of shape (1,) containing the loss
    """
    # print(f"\n\nSCores {scores}")
    # print(f"Y {y}")
    # print(f"Candidate ids {candidate_ids}")
    candidate_ids = list(candidate_ids)
    target_candidate_id = y.item()
    target_index = candidate_ids.index(target_candidate_id)
    target_index_tensor = torch.tensor([target_index]).to(y.device)
    loss = F.cross_entropy(scores.unsqueeze(0), target_index_tensor)

    return loss