"""Style-conditioned pix2pix-style cGAN: U-Net generator + PatchGAN discriminator (128x128).

The learned style embedding (nn.Embedding(3, emb_dim)) enters BOTH networks:
  G: broadcast as extra input channels AND concatenated at the 1x1 bottleneck,
  D: broadcast as extra input channels next to (photo, sketch).
"""
import torch
import torch.nn as nn


def _down(i, o, norm=True):
    layers = [nn.Conv2d(i, o, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(o))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _up(i, o, drop=0.0):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.Dropout(drop),
                         nn.ReLU(True))


class UNetGenerator(nn.Module):
    def __init__(self, base=64, emb_dim=16, dropout=0.3, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, emb_dim)
        self.e1, self.e2, self.e3 = _down(3 + emb_dim, b, False), _down(b, 2 * b), _down(2 * b, 4 * b)
        self.e4, self.e5, self.e6 = _down(4 * b, 8 * b), _down(8 * b, 8 * b), _down(8 * b, 8 * b)
        self.e7 = _down(8 * b, 8 * b, False)  # 1x1 bottleneck
        self.u1 = _up(8 * b + emb_dim, 8 * b, dropout)
        self.u2, self.u3 = _up(16 * b, 8 * b, dropout), _up(16 * b, 8 * b, dropout)
        self.u4, self.u5, self.u6 = _up(16 * b, 4 * b), _up(8 * b, 2 * b), _up(4 * b, b)
        self.out = nn.Sequential(nn.ConvTranspose2d(2 * b, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.emb(style)  # (B, emb_dim)
        x = torch.cat([x, e[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        h = torch.cat([e7, e[:, :, None, None]], 1)
        d = torch.cat([self.u1(h), e6], 1)
        d = torch.cat([self.u2(d), e5], 1)
        d = torch.cat([self.u3(d), e4], 1)
        d = torch.cat([self.u4(d), e3], 1)
        d = torch.cat([self.u5(d), e2], 1)
        d = torch.cat([self.u6(d), e1], 1)
        return self.out(d)


class PatchDiscriminator(nn.Module):
    def __init__(self, base=64, emb_dim=16, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            _down(3 + 1 + emb_dim, b, False), _down(b, 2 * b), _down(2 * b, 4 * b),
            nn.Conv2d(4 * b, 8 * b, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * b), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * b, 1, 4, 1, 1))

    def forward(self, photo, sketch, style):
        e = self.emb(style)[:, :, None, None].expand(-1, -1, photo.size(2), photo.size(3))
        return self.net(torch.cat([photo, sketch, e], 1))  # (B,1,14,14) patch logits


def build_gan(p):
    return (UNetGenerator(p["base"], p["emb_dim"], p["dropout"]), PatchDiscriminator(p["base"], p["emb_dim"]))
