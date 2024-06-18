import torch
from models.backbone import ResNetBackbone
from models.decoder import AttentionDecoder
from models.conclr import ConCLR
from utils.data_loader import get_data_loader

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = ResNetBackbone().to(device)
    decoder = AttentionDecoder(hidden_dim=512, num_classes=37).to(device)
    model = ConCLR(backbone, decoder).to(device)
    
    data_loader = get_data_loader('data/benchmarks', batch_size=32, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    evaluate()
    