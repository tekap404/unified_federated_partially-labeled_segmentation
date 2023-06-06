import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction
from monai.losses import DiceLoss
import math

class DiceBceMultilabelLoss(_Loss):
    def __init__(
        self,
        cfg,
        w_dice = 0.5,
        w_bce = 0.5,
        reduction = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.cfg = cfg
        self.w_dice = w_dice
        self.w_bce = w_bce

        self.dice_loss_single = DiceLoss(softmax=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=False, squared_pred=True, reduction="none")
        self.bce_loss_single = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01, include_background=True, batch=True, squared_pred=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.sup_weight = [0.1, 0.2, 0.3, 0.4, 1]

    def forward(self, pred, label, rce_flag=False, epoch=0, confidence_weight=None, pixel_flag=False):

        if confidence_weight != None:
            # deep supervision
            for i in range(5):
                dice_loss = torch.mean( confidence_weight * torch.mean( self.dice_loss_single(pred[:,i], label), dim=1 ).view(-1) )
                bce_loss = torch.mean( confidence_weight * torch.mean( self.bce_loss_single(pred[:,i], label).view(label.shape[0],-1), dim=1 ) )
                try:
                    loss = loss + self.sup_weight[i] * ( dice_loss * self.w_dice + bce_loss * self.w_bce )
                except:
                    loss = self.sup_weight[i] * ( dice_loss * self.w_dice + bce_loss * self.w_bce )
                # rbce loss
                if rce_flag == True:
                    rbce_loss = torch.mean( (confidence_weight * torch.mean( self.bce_loss_single(label, pred[:,i]).view(label.shape[0],-1), dim=1 ) ))
                    loss = loss + self.sup_weight[i] * rbce_loss * (self.cfg.rce_alpha * math.exp(-20*(1-epoch/self.cfg.epochs)) )
        else:
            for i in range(5):
                try:
                    loss = loss + self.sup_weight[i] * (self.dice_loss(pred[:,i], label) * self.w_dice + self.bce_loss(pred[:,i], label) * self.w_bce)
                except:
                    loss = self.sup_weight[i] * (self.dice_loss(pred[:,i], label) * self.w_dice + self.bce_loss(pred[:,i], label) * self.w_bce)
                if rce_flag == True:
                    rbce_loss = self.bce_loss(label, pred[:,i])
                    loss = loss + self.sup_weight[i] * rbce_loss * (self.cfg.rce_alpha * math.exp(-20*(1-epoch/self.cfg.epochs)) )

        return loss