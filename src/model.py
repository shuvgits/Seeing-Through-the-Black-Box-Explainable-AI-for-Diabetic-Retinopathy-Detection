"""
model.py
Diabetic retinopathy grading model using a pretrained EfficientNet-B4 backbone.
"""

import torch
import torch.nn as nn
import timm


class DRClassifier(nn.Module):
    """
    DR grading classifier with a pretrained EfficientNet-B4 backbone.

    Args:
        num_classes (int): Number of output classes (5 for DR grades 0-4)
        pretrained (bool): Load ImageNet pretrained weights
        dropout (float): Dropout probability before the classifier head
    """

    def __init__(self, num_classes=5, pretrained=True, dropout=0.3):
        super().__init__()

        # Load backbone without its original head
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,   # removes the default classifier
        )

        # Feature dimension from EfficientNet-B4
        in_features = self.backbone.num_features  # 1792

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)   # (B, 1792)
        logits = self.classifier(features)  # (B, num_classes)
        return logits


def build_model(num_classes=5, pretrained=True, dropout=0.3):
    """Convenience factory function."""
    return DRClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 5)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
