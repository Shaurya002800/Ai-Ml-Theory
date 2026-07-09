# CIFAR-10 Image Classifier

Classifying 10 object categories with transfer learning from ResNet18.
**Best val accuracy: X%** | [W&B Dashboard →](YOUR_WANDB_LINK_HERE)

## Results
| Model              | Val Acc | Epochs | Params  |
|--------------------|---------|--------|---------|
| Basic CNN (Day 3)  | ~67%    | 10     | ~220K   |
| CNN + BN + Dropout | ~75%    | 15     | ~250K   |
| ResNet18 (transfer)| X%      | 15     | ~11M    |

## Architecture
ResNet18 pretrained on ImageNet, fine-tuned in two phases:
- Phase 1: frozen backbone, train classifier head only (5 epochs)
- Phase 2: full network, backbone lr=1e-4, head lr=1e-3 (10 epochs)

## Key findings
- Transfer learning hit X% in 15 epochs vs ~67% training from scratch
- Cats and dogs were most commonly confused (similar features)
- Data augmentation reduced train/val gap from X% to Y%

## Usage
```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg
```

## What I'd do next
- Try ResNet50 or EfficientNet backbone
- Use test-time augmentation for better inference accuracy
- Deploy as a Gradio web app