import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Dice coefficient loss for evaluating overlap between predicted and ground truth masks.
        
        Parameters:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not applied in the model
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1):
        """
        Combined Dice and Binary Cross-Entropy loss.
        
        Parameters:
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not applied in the model
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Combined loss
        return bce_loss + dice_loss
