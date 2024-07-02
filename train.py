import argparse
import logging
import os

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

def get_optimizer(model, config):
    optimizer_config = config.optimizer
    optimizer_type = optimizer_config['type']
    if optimizer_type == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=optimizer_config['lr'], )
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'], betas=optimizer_config['args_betas'])
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_config['lr'])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    optimizer.load_state_dict(torch.load(config.model_checkpoint)['optimizer'])
    return optimizer

def get_scheduler(optimizer, config):
    scheduler_config = config.scheduler
    scheduler_type = scheduler_config['type']
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config['T_max'])
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    scheduler.load_state_dict(torch.load(config.model_checkpoint)['scheduler'])
    return scheduler

def eval_model(model, eval_data_loader, device):
    model.eval()
    total_eval_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for images, labels, view1, view2, labels1, labels2 in eval_data_loader:
            res_o = model(images.to(device))
            res1 = model(view1.to(device))
            res2 = model(view2.to(device))
            loss = total_loss(res_o['logits'], res1['logits'], res2['logits'], labels.to(device), labels1.to(device), labels2.to(device))
            total_eval_loss += loss.item()

            preds = res_o['logits'].argmax(dim=2)
            corrects += torch.all(preds == labels.to(device), axis=1).sum().item()

    return total_eval_loss / len(eval_data_loader), corrects / len(eval_data_loader) / eval_data_loader.batch_size

def save_checkpoint(model, optimizer, scheduler, target):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        }, target)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseVision(config).to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    writer = SummaryWriter(log_dir=config.global_workdir)

    train_loader = get_data_loader(config.dataset_root, config.dataset_train_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))
    eval_loader = get_data_loader(config.dataset_root, config.dataset_test_labels, batch_size=ifnone(config.dataset_test_batch_size, 64))

    checkpoints_dir = os.path.join(config.global_workdir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    step = 1
    for epoch in range(config.training_epochs):
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

            logging.info(f'Step {step}, Loss: {loss.item()}, Rec_Loss: {rec_loss.item()}, Clr_Loss: {clr_loss.item()}, Learning Rate: {scheduler.get_last_lr()[0]}')
            
            writer.add_scalar('Loss/total_loss', loss.item(), step)
            writer.add_scalar('Loss/rec_loss', rec_loss.item(), step)
            writer.add_scalar('Loss/clr_loss', clr_loss.item(), step)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], step)

            if step % config.training_save_iters == 0:
                save_checkpoint(model, optimizer, scheduler, os.path.join(checkpoints_dir, f'step_{step}.pth'))
                logging.info(f'Model saved at step {step}')

            if step % config.training_eval_iters == 0:
                eval_loss, acc = eval_model(model, eval_loader, device)
                writer.add_scalar('Evaluation/Loss', eval_loss, step)
                writer.add_scalar('Evaluation/Accuracy', acc, step)
                logging.info(f'Step {step}, Evaluation Loss: {eval_loss}, Accuracy: {acc}')
                save_checkpoint(model, optimizer, scheduler, os.path.join(checkpoints_dir, f'step_{step}_eval_{eval_loss}_acc_{acc}.pth'))
                logging.info(f'Model saved at step {step}')

            step += 1
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        save_checkpoint(model, optimizer, scheduler, os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth'))
        
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    config = Config(args.config)
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)
    train(config)
    