import logging
import torch.nn as nn

from models.attention import PositionAttention, Attention
from models.backbone import ResTranformer
from models.model import Model
from models.resnet import resnet45
from utils import ifnone

class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_d_model, 512)

        if config.model_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()
        
        if config.model_attention == 'position':
            mode = ifnone(config.model_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.model_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_checkpoint is not None:
            logging.info(f'Read model from {config.model_checkpoint}.')
            self.load(config.model_checkpoint)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision'}
