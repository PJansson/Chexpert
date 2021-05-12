import torch
from torch import nn
from torch.nn import functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, y_hat, y):
        return self.criterion(y_hat, y, weight=(y >= 0).float())


class LabelCorrelationAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, y_hat, y):
        loss = self.criterion(y_hat, y)
        loss += self.cross_label_dependency_loss(y_hat, y)
        return loss

    def cross_label_dependency_loss(self, y_hat, y):
        loss = torch.zeros(y.size(0), device=y.device)

        for i, (y_, y_hat_) in enumerate(zip(y, y_hat)):
            y0 = torch.nonzero(y_ == 0)
            y1 = torch.nonzero(y_)

            output = torch.exp((torch.sub(y_hat_[y0], y_hat_[y1][:, None]))).sum()

            num_comparisons = y0.size(0) * y1.size(0)
            if num_comparisons > 0:
                loss[i] = output.div(num_comparisons)

        return loss.mean()
