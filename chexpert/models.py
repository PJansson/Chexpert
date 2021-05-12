import timm
import torch
from torch import nn


class SingleViewModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            name, num_classes=num_classes, pretrained=pretrained, **kwargs
        )

    def forward(self, x):
        return self.model(x)


class MultiViewModel(nn.Module):
    def __init__(
        self, name, num_classes, max_before_pool=True, pretrained=True, **kwargs
    ):
        super().__init__()
        self.max_before_pool = max_before_pool

        self.model = timm.create_model(
            name, num_classes=0, global_pool="", pretrained=pretrained, **kwargs
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        bs, n, c, h, w = x.size()
        x = x.view(bs * n, c, h, w)
        x = self.model(x)

        if self.max_before_pool:
            x = x.view(bs, n, x.size(1), x.size(2), x.size(3))
            x = torch.max(x, 1)[0]
            x = self.pool(x).squeeze()
        else:
            x = self.pool(x).squeeze()
            x = x.view(bs, n, self.model.num_features)
            x = torch.max(x, 1)[0]

        x = self.linear(x)
        return x
