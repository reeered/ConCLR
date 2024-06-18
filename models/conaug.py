import torch
import torch.nn.functional as F

class ConAug:
    def __init__(self, transform):
        self.transform = transform
        pass
    
    def augment(self, images, labels, max_length: int=30):
        batch_size = images.size(0)
        perm1 = torch.randperm(batch_size)
        perm2 = torch.randperm(batch_size)
        view1 = self.transform(torch.cat((images, images[perm1]), dim=3))
        view2 = self.transform(torch.cat((images, images[perm2]), dim=3))
        labels1 = torch.cat((labels, labels[perm1]), dim=1)
        labels2 = torch.cat((labels, labels[perm2]), dim=1)
        F.pad(labels1, (0, max_length - labels1.size(1)))

        return view1, view2, labels1, labels2
    