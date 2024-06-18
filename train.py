import torch
import torch.optim as optim
from models.model_vision import BaseVision
from utils import get_data_loader
from losses import total_loss
from utils import Config, ifnone

def train():
    config = Config('configs/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseVision(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_loader = get_data_loader('data/ICDAR2013+2015/train_data', batch_size=ifnone(config.dataset_train_batch_size, 4))

    step = 0
    for epoch in range(10):
        model.train()
        for images, labels, view1, view2, labels1, labels2 in data_loader:
            # images = images.to(device)
            # view1, view2, labels1, labels2 = conaug.augment(images, labels)
            res_o = model(images.to(device))
            res1 = model(view1.to(device))
            res2 = model(view2.to(device))
            
            loss = total_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()}')
            step += 1
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}_{loss.item()}.pth')

if __name__ == '__main__':
    train()