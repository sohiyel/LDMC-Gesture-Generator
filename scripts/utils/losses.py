import torch
import torch.nn as nn

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred, motion_gt)
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : ], motion_gt[..., 4 : ])
        return loss
    
    