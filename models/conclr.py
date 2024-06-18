import torch.nn as nn
import torch

class ConCLR(nn.Module):
    def __init__(self, backbone, decoder, projection_dim=128, num_classes=37):
        super(ConCLR, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.projection = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection(features)
        return projections