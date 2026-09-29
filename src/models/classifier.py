"""4-class corruption-type classifier: clean / salt_pepper / blur / occlusion (src.data.corruptions.TYPES order)."""
import torch.nn as nn

CHANNEL_OPTIONS = {
    "c16": (16, 32, 64, 128),
    "c32": (32, 64, 128, 256),
    "c48": (48, 96, 192, 384),
}


def conv_block(cin, cout, stride=1, p=0.0):
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if p > 0:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class CorruptionClassifier(nn.Module):
    """Conv stem (stride-2 x4, 128->8) -> global average pool -> dropout -> linear(4)."""

    def __init__(self, channels=(32, 64, 128, 256), dropout=0.2, n_classes=4):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers += [conv_block(cin, c, stride=2, p=dropout * 0.5), conv_block(c, c)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_classes))
        self.channels, self.dropout, self.n_classes = channels, dropout, n_classes

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_classifier(params: dict) -> CorruptionClassifier:
    return CorruptionClassifier(CHANNEL_OPTIONS[params["channels"]], params["dropout"])
