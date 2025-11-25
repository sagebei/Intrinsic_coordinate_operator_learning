import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

class H1LossMetric(Metric):
    def __init__(self):
        super().__init__()
        # Sum of batch losses
        self.add_state("sum_h1_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # Number of samples accumulated
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def gradient(x: torch.Tensor):
        dx = x[..., :, 1:] - x[..., :, :-1]    # horizontal gradient
        dy = x[..., 1:, :] - x[..., :-1, :]    # vertical gradient
        return dx, dy

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Compute L2 loss on values
        l2_loss = F.mse_loss(preds, targets, reduction="mean")

        # Compute gradients
        pred_dx, pred_dy = self.gradient(preds)
        target_dx, target_dy = self.gradient(targets)

        # Compute L2 loss on gradients
        grad_loss_x = F.mse_loss(pred_dx, target_dx, reduction="mean")
        grad_loss_y = F.mse_loss(pred_dy, target_dy, reduction="mean")
        grad_loss = grad_loss_x + grad_loss_y

        # H1 loss for this batch
        batch_h1_loss = l2_loss + grad_loss

        # Accumulate sum weighted by batch size
        batch_size = preds.shape[0]
        self.sum_h1_loss += batch_h1_loss * batch_size
        self.total += batch_size

    def compute(self):
        # Return average loss over all batches accumulated
        return self.sum_h1_loss / self.total

    def reset(self) -> None:
        # Reset states in-place to keep device and dtype consistent
        self.sum_h1_loss.zero_()
        self.total.zero_()


class H1Loss(nn.Module):
    def __init__(self):
        super(H1Loss, self).__init__()

    @staticmethod
    def gradient(x):
        """
        Computes gradients ∂x/∂u and ∂x/∂v using finite differences.
        Input: x of shape (B, 1, H, W) or (B, H, W)
        Returns: dx, dy of shape (B, 1, H, W−1) and (B, 1, H−1, W)
        """
        if x.ndim == 3:
            # Add channel dimension if missing
            x = x.unsqueeze(1)

        dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # horizontal gradient (W axis)
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # vertical gradient (H axis)
        return dx, dy

    def forward(self, preds, targets):
        """
        Compute H1 loss = L2 loss + gradient L2 loss
        preds, targets: shape (B, 1, H, W) or (B, H, W)
        Returns: scalar tensor
        """
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            targets = targets.unsqueeze(1)

        # L2 loss
        l2_loss = F.mse_loss(preds, targets)

        # Gradient loss
        pred_dx, pred_dy = self.gradient(preds)
        target_dx, target_dy = self.gradient(targets)

        grad_loss_x = F.mse_loss(pred_dx, target_dx)
        grad_loss_y = F.mse_loss(pred_dy, target_dy)

        grad_loss = grad_loss_x + grad_loss_y

        # print(l2_loss.item(), grad_loss.item())
        return l2_loss + grad_loss


class RelativeH1Loss(nn.Module):
    def __init__(self, eps=1e-8):
        """
        Relative H1 loss:
        (||pred - target||^2 + ||∇pred - ∇target||^2) / (||target||^2 + ||∇target||^2 + eps)
        """
        super().__init__()
        self.eps = eps

    @staticmethod
    def gradient(x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, preds, targets):
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
            targets = targets.unsqueeze(1)

        # L2 terms
        diff = preds - targets
        l2_num = torch.sum(diff ** 2)
        l2_den = torch.sum(targets ** 2)

        # Gradient terms
        pred_dx, pred_dy = self.gradient(preds)
        target_dx, target_dy = self.gradient(targets)

        grad_diff_x = pred_dx - target_dx
        grad_diff_y = pred_dy - target_dy

        grad_num = torch.sum(grad_diff_x ** 2) + torch.sum(grad_diff_y ** 2)
        grad_den = torch.sum(target_dx ** 2) + torch.sum(target_dy ** 2)

        numerator = l2_num + grad_num
        denominator = l2_den + grad_den + self.eps

        rel_h1_loss = numerator / denominator

        return rel_h1_loss


def total_variation_loss(x):
    """
    Computes TV loss: encourages piecewise smoothness.
    Input: x shape (N, C, H, W)
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def laplacian_loss(x):
    """
    Computes Laplacian regularization loss: penalizes curvature.
    Input: x shape (N, C, H, W)
    """
    # Laplacian kernel
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32, device=x.device)
    kernel = kernel.view(1, 1, 3, 3)  # shape (1, 1, 3, 3)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)  # shape (C, 1, 3, 3)

    lap = F.conv2d(x, kernel, padding=1, groups=x.size(1))
    return torch.mean(lap ** 2)


if __name__ == "__main__":
    criterion = H1Loss(dx=1/64, dy=1/64, reduction='mean')

    pred = torch.randn(8, 1, 64, 64, requires_grad=True)
    target = torch.randn(8, 1, 64, 64)

    loss = criterion(pred, target)
    loss.backward()

    print(loss.item())