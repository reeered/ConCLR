import argparse
import logging

import torch
from tqdm import tqdm

from models.model_vision import BaseVision
from utils import get_data_loader, Config, Logger, ifnone, CharsetMapper

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model for vision task')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file (default: configs/config.yaml)')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (if not provided, will use config value)')
    parser.add_argument('--conclr_checkpoint', type=str, help='Path to conclr model checkpoint')
    args = parser.parse_args()
    return args

def calculate_accuracy(logits, labels):
    preds = logits.argmax(dim=2)
    corrects = torch.all(preds == labels, axis=1).sum().item()
    total = labels.size(0)
    charset = CharsetMapper('data/charset_36.txt')
    for pred, label in zip(preds, labels):
        pred = charset.get_text(pred).replace('\u2591', ' ')
        label = charset.get_text(label).replace('\u2591', ' ')
        logging.info(f'Pred: {pred}, Label: {label}')
    acc = corrects / total
    print(f'Corrects: {corrects}, Total: {total}, Acc: {acc}')
    return corrects, acc

def eval_model(model_baseline, model_conclr, eval_data_loader, device, num_batches=20000):
    model_baseline.eval()
    model_conclr.eval()
    total_corrects_baseline = 0
    total_corrects_conclr = 0
    acc_baseline = 0
    acc_conclr = 0
    n = 0
    with torch.no_grad():
        for images, labels in tqdm(eval_data_loader):
            if n == num_batches:
                break
            res_o_baseline = model_baseline(images.to(device))
            res_o_conclr = model_conclr(images.to(device))

            corrects, acc = calculate_accuracy(res_o_baseline['logits'], labels.to(device))
            total_corrects_baseline += corrects
            acc_baseline += acc

            corrects, acc = calculate_accuracy(res_o_conclr['logits'], labels.to(device))
            total_corrects_conclr += corrects
            acc_conclr += acc
            n += 1
    # acc_baseline /= len(eval_data_loader)
    # acc_conclr /= len(eval_data_loader)
    acc_baseline /= n
    acc_conclr /= n
    return acc_baseline, acc_conclr

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: checkpoint
    if args.checkpoint:
        config.model_checkpoint = args.checkpoint
    conclr_checkpoint = args.conclr_checkpoint

    model_baseline = BaseVision(config).to(device)
    model_conclr = BaseVision(config).to(device)
    model_conclr.load_state_dict(torch.load(conclr_checkpoint)['model'])

    eval_loader = get_data_loader(config.dataset_root, config.dataset_test_labels, batch_size=ifnone(config.dataset_test_batch_size, 16), conaug=False)

    acc_baseline, acc_conclr = eval_model(model_baseline, model_conclr, eval_loader, device)
    logging.info(f'Evaluation Accuracy:\nBaseline: {acc_baseline * 100:.2f}%\nConCLR:  {acc_conclr * 100:.2f}%')

if __name__ == '__main__':
    args = parse_args()
    config = Config(args.config)
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)

    main(config, args)
    