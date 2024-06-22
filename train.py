import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.model_vision import BaseVision
from utils import get_data_loader
from losses import total_loss, recognition_loss, contrastive_loss
from utils import Config, ifnone, Logger

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

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseVision(config).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.SGD(model.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    writer = SummaryWriter(log_dir=config.global_workdir)

    train_loader = get_data_loader(config.dataset_root, config.dataset_train_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))
    eval_loader = get_data_loader(config.dataset_root, config.dataset_test_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))

    step = 1
    for epoch in range(10):
        model.train()
        for images, labels, view1, view2, labels1, labels2 in train_loader:
            res_o = model(images.to(device))
            res1 = model(view1.to(device))
            res2 = model(view2.to(device))
        
            rec_loss = recognition_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            clr_loss = contrastive_loss(res1['logits'], res2['logits'], labels1.to(device), labels2.to(device))
            
            lambda_ = 0.2
            loss = rec_loss + lambda_ * clr_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logging.info(f'Step {step+1}, Loss: {loss.item()}, Rec_Loss: {rec_loss.item()}, Clr_Loss: {clr_loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}')
            
            writer.add_scalar('Loss/total_loss', loss.item(), step)
            writer.add_scalar('Loss/rec_loss', rec_loss.item(), step)
            writer.add_scalar('Loss/clr_loss', clr_loss.item(), step)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], step)

            if step % 2000 == 0:
                torch.save(model.state_dict(), f'checkpoints/model_step_{step+1}.pth')
                logging.info(f'Model saved at step {step+1}')

            if step % 10000 == 0:
                eval_loss = eval_model(model, eval_loader, device)
                logging.info(f'Step {step+1}, Evaluation Loss: {eval_loss}')
                torch.save(model.state_dict(), f'checkpoints/model_step_{step+1}.pth')
                logging.info(f'Model saved at step {step+1}')

            step += 1
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save(model.state_dict(), f'checkpoints/model_{epoch+1}_{loss.item()}.pth')
        
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    config = Config(args.config)
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)
    train(args, config)
    