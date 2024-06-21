import argparse
import time

import torch
import torch.optim as optim
from models.model_vision import BaseVision
from utils import get_data_loader
from losses import total_loss, recognition_loss, contrastive_loss
from utils import Config, ifnone

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for vision task')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file (default: configs/config.yaml)')
    args = parser.parse_args()
    return args

def eval_model(model, eval_data_loader, device):
    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for images, labels, view1, view2, labels1, labels2 in eval_data_loader:
            res_o = model(images.to(device))
            res1 = model(view1.to(device))
            res2 = model(view2.to(device))
            loss = total_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            total_eval_loss += loss.item()
    return total_eval_loss / len(eval_data_loader)

def train():
    args = parse_args()
    config = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseVision(config).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # data_loader = get_data_loader('data/ICDAR2013+2015/train_data', batch_size=ifnone(config.dataset_train_batch_size, 4))

    train_loader = get_data_loader(config.dataset_root, config.dataset_train_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))
    eval_loader = get_data_loader(config.dataset_root, config.dataset_test_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))

    step = 1
    for epoch in range(10):
        model.train()
        for images, labels, view1, view2, labels1, labels2 in train_loader:
            # images = images.to(device)
            # view1, view2, labels1, labels2 = conaug.augment(images, labels)
            res_o = model(images.to(device))
            res1 = model(view1.to(device))
            res2 = model(view2.to(device))
        
            rec_loss = recognition_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            clr_loss = contrastive_loss(res1['logits'], res2['logits'], labels1.to(device), labels2.to(device))
            
            lambda_ = 0.2
            loss = rec_loss + lambda_ * clr_loss
            
            # loss = total_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f'Step {step+1}, Loss: {loss.item()}, Rec_Loss: {rec_loss.item()}, Clr_Loss: {clr_loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}')
            
            if step % 500 == 0:
                eval_loss = eval_model(model, eval_loader, device)
                print(f'Step {step+1}, Evaluation Loss: {eval_loss}')
                torch.save(model.state_dict(), f'checkpoints/model_step_{step+1}.pth')
                print(f'Model saved at step {step+1}')

            step += 1
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}_{loss.item()}.pth')

if __name__ == '__main__':
    train()
    