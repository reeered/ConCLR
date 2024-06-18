import torch.nn as nn
import torch

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(AttentionDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.fc(attn_output)