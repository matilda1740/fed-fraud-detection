"""
    Deep FeedForward Neural Network that uses Binary Classification for Fraud Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.main(x)
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs.squeeze(), targets.float())
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1-pt)**self.gamma * bce_loss).mean()
    