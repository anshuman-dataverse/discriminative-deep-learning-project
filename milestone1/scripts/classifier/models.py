import torch.nn as nn
from torchvision import models

MODEL_NAMES = ["SimpleCNN", "ResNet18", "MobileNetV3", "EfficientNetB0"]
DEFAULTS = {
    "SimpleCNN": {"lr": 1e-3, "epochs": 40},
    "ResNet18": {"lr": 3e-4, "epochs": 15},
    "MobileNetV3": {"lr": 3e-4, "epochs": 15},
    "EfficientNetB0": {"lr": 3e-4, "epochs": 15},
}


class SimpleCNN(nn.Module):
    """Baseline trained from scratch: 4 x (2 x conv-BN-ReLU, max-pool), GAP, dropout, linear."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "SimpleCNN":
        return SimpleCNN(num_classes)
    if name == "ResNet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "MobileNetV3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
