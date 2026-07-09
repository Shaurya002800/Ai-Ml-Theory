import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import sys

classes = ['plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def load_model(path='best_resnet18.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,10)
    )
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)   # add batch dim

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = torch.argmax(probs).item()

    print(f"Prediction:  {classes[pred]}")
    print(f"Confidence:  {probs[pred].item():.1%}")
    print("\nTop 3:")
    top3 = torch.topk(probs, 3)
    for prob, idx in zip(top3.values, top3.indices):
        print(f"  {classes[idx.item()]:8s} {prob.item():.1%}")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    model      = load_model()
    predict(image_path, model)