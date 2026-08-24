import torch.nn as nn
from hri_gnn.config.const import ENTITY_ATTRIBUTES

class RavenCnnEncoder(nn.Module):
    def __init__(self, output_shape:int = 64):
        super().__init__()
        self.conv_block = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),     
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
          nn.Flatten()
        )

        self.fc_latent = nn.Sequential( nn.Linear(in_features=64*5*5, out_features=output_shape), nn.ReLU())

        self.head_classifier_type = nn.Linear(in_features=output_shape, out_features=len(ENTITY_ATTRIBUTES["type"]))
        self.head_classifier_size = nn.Linear(in_features=output_shape, out_features=len(ENTITY_ATTRIBUTES["size"]))
        self.head_classifier_color = nn.Linear(in_features=output_shape, out_features=len(ENTITY_ATTRIBUTES["color"]))
        self.head_classifier_angle = nn.Linear(in_features=output_shape, out_features=len(ENTITY_ATTRIBUTES["angle"]))
         
    def forward(self, x):
        """
        Forward pass through the network
        """
        # Convolutional Feature Extraction
        x = self.conv_block(x)
        # print(f"conv_block out: {x.shape}")

        # Latent Feature Learning
        latent_vector = self.fc_latent(x)
        # print(f"fc_latent out: {latent_vector.shape}")
        
        # Classification Heads
        pred_type_logits = self.head_classifier_type(latent_vector)
        # print(f"head_classifier_type out: {pred_type_logits.shape}")
        pred_size_logits = self.head_classifier_size(latent_vector)
        # print(f"head_classifier_size out: {pred_size_logits.shape}")
        pred_color_logits = self.head_classifier_color(latent_vector)
        # print(f"head_classifier_color out: {pred_color_logits.shape}")
        pred_angle_logits = self.head_classifier_angle(latent_vector)
        # print(f"head_classifier_angle out: {pred_angle_logits.shape}")
        return {"latent_vector":latent_vector, "pred_type_logits":pred_type_logits, "pred_size_logits":pred_size_logits, "pred_color_logits":pred_color_logits, "pred_angle_logits":pred_angle_logits}
