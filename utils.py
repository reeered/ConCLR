import logging
import os
import yaml

from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def ifnone(a, b):
    return b if a is None else a

class CharsetMapper(object):
    """A simple class to map ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    """

    def __init__(self,
                 filename='',
                 max_length=30,
                 null_char=u'\u2591'):
        """Creates a lookup table.

        Args:
          filename: Path to charset file which maps characters to ids.
          max_sequence_length: The max length of ids and string.
          null_char: A unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.
        """
        self.null_char = null_char
        self.max_length = max_length

        self.label_to_char = self._read_charset(filename)
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)
 
    def _read_charset(self, filename):
        """Reads a charset definition from a tab separated text file.

        Args:
          filename: a path to the charset file.

        Returns:
          a dictionary with keys equal to character codes and values - unicode
          characters.
        """
        import re
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                m = pattern.match(line)
                assert m, f'Incorrect charset file. line #{i}: {line}'
                label = int(m.group(1)) + 1
                char = m.group(2)
                charset[label] = char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, '')

    def get_text(self, labels, length=None, padding=True, trim=False):
        """ Returns a string corresponding to a sequence of character ids.
        """
        length = length if length else self.max_length
        labels = [l.item() if isinstance(l, Tensor) else int(l) for l in labels]
        if padding:
            labels = labels + [self.null_label] * (length-len(labels))
        text = ''.join([self.label_to_char[label] for label in labels])
        if trim: text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length

        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return '0123456789'

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                valid_chars.append(c)
        return ''.join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


class Logger(object):
    _handle = None
    _root = None

    @staticmethod
    def init(output_dir, name, phase):
        format = '[%(asctime)s %(filename)s:%(lineno)d %(levelname)s {}] ' \
                '%(message)s'.format(name)
        logging.basicConfig(level=logging.INFO, format=format)

        try: os.makedirs(output_dir)
        except: pass
        config_path = os.path.join(output_dir, f'{phase}.txt')
        Logger._handle = logging.FileHandler(config_path)
        Logger._root = logging.getLogger()

    @staticmethod
    def enable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.addHandler(Logger._handle)

    @staticmethod
    def disable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.removeHandler(Logger._handle)


class Config(object):

    def __init__(self, config_path, host=True):
        def __dict2attr(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    __dict2attr(v, f'{prefix}{k}_')
                else:
                    if k == 'phase':
                        assert v in ['train', 'test']
                    self.__setattr__(f'{prefix}{k}', v)

        assert os.path.exists(config_path), '%s does not exists!' % config_path
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        with open('configs/template.yaml') as file:
            default_config_dict = yaml.load(file, Loader=yaml.FullLoader)
        __dict2attr(default_config_dict)
        __dict2attr(config_dict)
        self.global_workdir = os.path.join(self.global_workdir, self.global_name)

    def __getattr__(self, item):
        attr = self.__dict__.get(item)
        if attr is None:
            attr = dict()
            prefix = f'{item}_'
            for k, v in self.__dict__.items():
                if k.startswith(prefix):
                    n = k.replace(prefix, '')
                    attr[n] = v
            return attr if len(attr) > 0 else None
        else:
            return attr

    def __repr__(self):
        str = 'ModelConfig(\n'
        for i, (k, v) in enumerate(sorted(vars(self).items())):
            str += f'\t({i}): {k} = {v}\n'
        str += ')'
        return str
    

class ConCLRDataset(Dataset):
    def __init__(self, data_dir: str, labels_path: str, transform=None, charset_path: str='data/charset_36.txt', max_length: int = 26):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = [], []
        with open(labels_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 2
            img_file, _ = parts
            label = img_file.split('_')[1].lower()
            self.images.append(os.path.join(data_dir, img_file))
            self.labels.append(label)

        self.charset = CharsetMapper(charset_path, max_length=max_length)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # img_path = os.path.join(self.data_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except OSError as e:
            #TODO: 数据集有部分图像损坏
            logging.error(e)
            logging.error(f'Broken image: {img_path}')
            image = Image.open(self.images[idx+1]).convert('RGB')
            label = self.charset.get_labels(self.labels[idx+1], padding=False)
            label = torch.tensor(label).to(dtype=torch.long)
            return self.transform(image), label
            # raise Exception(f'Broken image: {img_path}')

        # original text
        text = self.labels[idx]
        # convert text to label
        label = self.charset.get_labels(text, padding=False)
        label = torch.tensor(label).to(dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loader(data_dir: str, labels_path: str, batch_size: int, max_length: int=26, charset_path: str = 'data/charset_36.txt', shuffle=True, conaug=True):
    """Get ConCLR Dataloader.

    Args:
        data_dir (str): Directory of images.
        labels_path (str): Path to labels file.
        batch_size (int): Batch size.
        max_length (int, optional): Max length of labels. Defaults to 26.
        charset_path (str, optional): Path to charset file. Defaults to 'data/charset_36.txt'.
        shuffle (bool, optional): Shuffle data. Defaults to True.
        conaug (bool, optional): Use ConCLR augmentation. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Dataloader
    """
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ConCLRDataset(data_dir, labels_path, transform, max_length=max_length, charset_path=charset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_wrapper(max_length, conaug))

def collate_wrapper(max_length, conaug):
    def collate_fn(batch):
        """Concat images and labels in a batch."""
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        if conaug:
            bs = images.size(0)
            perm1 = torch.randperm(bs)
            perm2 = torch.randperm(bs)

            labels_pad = []
            labels1 = []
            labels2 = []
            for i in range(bs):
                # label padding
                label_cat = torch.cat((labels[i], labels[perm1[i]]))
                labels1.append(F.pad(label_cat, (0, max_length - label_cat.size(0))))
                label_cat = torch.cat((labels[i], labels[perm2[i]]))
                labels2.append(F.pad(label_cat, (0, max_length - label_cat.size(0))))
                labels_pad.append(F.pad(labels[i], (0, max_length - labels[i].size(0))))

            # image scaling
            view1 = F.interpolate(torch.cat((images, images[perm1]), dim=3), (32, 128))
            view2 = F.interpolate(torch.cat((images, images[perm2]), dim=3), (32, 128))

            labels = torch.stack(labels_pad, 0)
            labels1 = torch.stack(labels1, 0)
            labels2 = torch.stack(labels2, 0)

            return images, labels, view1, view2, labels1, labels2
        else:
            labels = [F.pad(label, (0, max_length - label.size(0))) for label in labels]
            labels = torch.stack(labels, 0)
            return images, labels
    return collate_fn
