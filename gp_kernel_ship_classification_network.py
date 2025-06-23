import os
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# class GPKernelShipClassificationNetwork(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden_dim=32):
#         super().__init__()
        
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )
        
#     def forward(self, x):
#         return self.net(x)

class GPKernelShipClassificationNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # More complex architecture with regularization
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Add dropout for regularization
            current_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # More complex architecture with regularization
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Add dropout for regularization
            current_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)