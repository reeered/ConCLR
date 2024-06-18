import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models.model_vision import BaseVision
from models.decoder import AttentionDecoder
from models.conaug import ConAug
from models.conclr import ConCLR
from utils import get_data_loader
from losses import total_loss
from utils import Config

def train():
    config = Config('configs/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseVision(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 减小学习率
    
    data_loader = get_data_loader('data/ICDAR2013+2015/train_data', batch_size=32)
    
    for epoch in range(10):
        model.train()
        for images, labels, view1, view2, labels1, labels2 in data_loader:
            # images = images.to(device)
            # view1, view2, labels1, labels2 = conaug.augment(images, labels)
            res_o = model(images)
            res1 = model(view1)
            res2 = model(view2)
            
            loss = total_loss(res_o['logits'], res1['logits'], res2['logits'], labels, labels1, labels2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    train()