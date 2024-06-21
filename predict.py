import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.model_vision import BaseVision
from utils import get_data_loader, ifnone, Config, CharsetMapper

def parse_args():
    parser = argparse.ArgumentParser(description='Predict using a trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file (default: configs/config.yaml)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best-pretrain-vision-model.pth', help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='predictions.txt', help='Path to save the prediction results')
    args = parser.parse_args()
    return args

def load_model(config, checkpoint, device='cpu', strict=True):
    model = BaseVision(config).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if isinstance(device, int): 
        device = torch.device('cuda', device)
    assert os.path.isfile(checkpoint)
    state = torch.load(checkpoint, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_


def save_predictions(predictions, output_path):
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def main():
    args = parse_args()
    config = Config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config, args.checkpoint, device)
    
    test_loader = get_data_loader(config.dataset_root, config.dataset_test_labels, batch_size=ifnone(config.dataset_train_batch_size, 16))

    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    
    predictions = []

    data_dir = config.dataset_root
    with open(config.dataset_train_labels, 'r') as f:
        lines = f.readlines()

    with torch.no_grad():
        for line in tqdm(lines):
            img_path, label = line.strip().split('\t')
            img = Image.open(os.path.join(data_dir, img_path)).convert('RGB')
            img = preprocess(img, config.dataset_image_width, config.dataset_image_height)
            img = img.to(device)
            res = model(img)
            pt_text, _, __ = postprocess(res, charset, config.model_eval)
            if pt_text == label.lower():
                pt_text += "GOOD"
            predictions.append(f'{pt_text}\t{label}')

        # for images, _, _, _, _, _ in test_loader:
        #     res = model(images.to(device))
        #     pt_text, _, __ = postprocess(res, charset, config.model_eval)
        #     predictions.append(pt_text)
    save_predictions(predictions, args.output_path)

    print(f"Predictions saved to {args.output_path}")

if __name__ == '__main__':
    main()