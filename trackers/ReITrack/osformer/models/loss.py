import torch
import torch.nn as nn
from mggan_motion.models.motion_modules.losses import IOUloss


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.iou_loss_fn = IOUloss(reduction="sum")
        self.mse_loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, pred, gt):
        """

            :param pred: (N, 4), xywh
            :param gt: (N, 4), xywh
            :return:
        """
        iou_loss = self.iou_loss_fn(pred, gt)
        mse_loss = self.mse_loss_fn(pred, gt)
        loss = iou_loss + mse_loss
        return loss, iou_loss, mse_loss


class IoULoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(IoULoss, self).__init__()
        self.iou_loss_fn = IOUloss(reduction=reduction)

    def forward(self, pred, gt):
        """

            :param pred: (N, 4), xywh
            :param gt: (N, 4), xywh
            :return:
        """
        iou_loss = self.iou_loss_fn(pred, gt)
        return iou_loss, iou_loss, torch.zeros_like(iou_loss)


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred, gt):
        """

            :param pred: (N, 4), xywh
            :param gt: (N, 4), xywh
            :return:
        """
        mse_loss = self.mse_loss_fn(pred, gt)
        return mse_loss, torch.zeros_like(mse_loss), mse_loss
