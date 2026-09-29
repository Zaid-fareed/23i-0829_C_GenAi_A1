"""Convolutional denoising autoencoder (used by Task 1 and, unchanged, by the Task 2/3 specialists).

Encoder : 4 x [Conv3x3 stride2 -> BN -> ReLU, Conv3x3 -> BN -> ReLU]        128 -> 8 spatial
Bottleneck: 1x1 conv to `bottleneck_dim` channels (LINEAR latent, shape bottleneck_dim x 8 x 8), Dropout2d, 1x1 conv back
Decoder : 4 x [Upsample x2 (nearest) -> Conv3x3 -> BN -> ReLU, Conv3x3 -> BN -> ReLU -> Dropout2d], Conv3x3 -> Sigmoid
NO skip connections: everything must pass through the latent (bottleneck_dim*8*8 numbers, e.g. 64*64 = 4096 vs 49152 pixels).
"""
import torch
import torch.nn as nn

CHANNEL_OPTIONS = {  # 4 stages each; Optuna chooses the key
    "c16": (16, 32, 64, 128),
    "c32": (32, 64, 128, 256),
    "c48": (48, 96, 192, 384),
}


def conv_block(cin, cout, stride=1, p=0.0):
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if p > 0:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), bottleneck_dim=64, dropout=0.1):
        super().__init__()
        channels = tuple(channels)
        enc, cin = [], 3
        for c in channels:
            enc += [conv_block(cin, c, stride=2), conv_block(c, c)]
            cin = c
        self.encoder = nn.Sequential(*enc)
        self.to_latent = nn.Conv2d(cin, bottleneck_dim, 1)
        self.latent_drop = nn.Dropout2d(dropout)
        self.from_latent = nn.Sequential(nn.Conv2d(bottleneck_dim, cin, 1), nn.ReLU(inplace=True))
        dec = []
        for c_out in list(reversed(channels))[1:] + [channels[0]]:
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(cin, c_out),
                    conv_block(c_out, c_out, p=dropout)]
            cin = c_out
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Sequential(nn.Conv2d(cin, 3, 3, 1, 1), nn.Sigmoid())
        self.channels, self.bottleneck_dim, self.dropout = channels, bottleneck_dim, dropout

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        return self.out(self.decoder(self.from_latent(self.latent_drop(z))))

    def forward(self, x):
        return self.decode(self.encode(x))


def build_ae(params: dict) -> ConvAutoencoder:
    """params: {'channels': key of CHANNEL_OPTIONS, 'bottleneck_dim': int, 'dropout': float}"""
    return ConvAutoencoder(CHANNEL_OPTIONS[params["channels"]], params["bottleneck_dim"], params["dropout"])
