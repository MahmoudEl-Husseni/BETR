# ------------------------------------------
# This file is part of metrics.
# File: ade.py

# Autor: Mahmoud ElHusseni
# Created on 2024/01/26.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------
import torch 
import numpy as np
from torchmetrics import Metric


def mean_displacement_error(y_pred, y):
    """
        Compute the final displacement error between the ground truth and the prediction.
    """
    error_ls = torch.norm(y_pred - y[:, None], dim=-1) # shape -> bs * K
    error_ls = error_ls.mean(dim=-1)
    return error_ls # shape -> bs * K


class MinMDE(Metric):
    """
    Compute the Mean displacement error between the ground truth and the prediction.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        """Update metric.

        Args:
            pred: predicted trajectories [N, K, T, 2]
            target: ground truth trajectories [N, T, 2]
        """
        mde = mean_displacement_error(pred, target)
        min_id = mde.argmin(dim=-1)
        self.sum += mde[torch.arange(len(min_id)), min_id].sum()
        self.count += pred.size(0)


    def compute(self):
        """
        Computes mean displacement error over state.
        
        Returns:
            Tensor: mean displacement error.
        """
        return self.sum / self.count