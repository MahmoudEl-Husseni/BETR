# ------------------------------------------
# This file is part of metrics.
# File: mr.py

# Autor: Mahmoud ElHusseni
# Created on 2024/01/26.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------
from config import *

import torch
import numpy as np
from torchmetrics import Metric

def compute_missrate(y_pred, y_true, heading, T, comb='avg') -> int:
    """
    Compute the miss rate between the ground truth and the prediction.

                        λlat    λlon
        T=3 seconds     1       2
        T=5 seconds     1.8     3.6
        T=8 seconds     3       6
    
    """
    R : np.ndarray = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    if comb == 'avg':
      MR = np.zeros((len(y_pred), T))
    elif comb=='min': 
      MR = np.ones((len(y_pred), T))
    for i in range(y_pred.shape[1]):
        err = y_true - y_pred[:, i]

        err = np.matmul(err, R) # shape -> bs * T * 2
        lat = [1, 1.8, 3]
        lon = [2, 3.6, 6]
        samples = [slice(30), slice(30, 50), slice(50, T)]

        for j in range(len(lat)):
            if comb=='min': 
              MR[:, samples[j]] = np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]) & (MR[:, samples[j]]==1), 1, 0)
            elif comb=='avg': 
              MR[:, samples[j]] += np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]), 1, 0)
        
        if comb=='avg': 
          MR = MR / y_pred.shape[1]
    return MR

def compute_mr(pred, gt, T, comb='avg'):
    """
    Compute the miss rate between the ground truth and the prediction.

                        λlat    λlon
        T=3 seconds     1       2
        T=5 seconds     1.8     3.6
        T=8 seconds     3       6
    
    """
    if comb=='avg':
      MR = np.zeros((len(pred), T))
    elif comb=='min':
      MR = np.ones((len(pred), T))
    
    err = gt - pred

    lat = [1, 1.8, 3]
    lon = [2, 3.6, 6]
    samples = [slice(30), slice(30, 50), slice(50, T)]

    for j in range(len(lat)):
        if comb=='min': 
          MR[:, samples[j]] = np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]) & (MR[:, samples[j]]==1), 1, 0)
        elif comb=='avg': 
          MR[:, samples[j]] += np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]), 1, 0)

    return MR


   
class MissRate(Metric):
    """
    Compute the miss rate between the ground truth and the prediction.

                        λlat    λlon
        T=3 seconds     1       2
        T=5 seconds     1.8     3.6
        T=8 seconds     3       6
    
    """
    def __init__(self, T=N_FUTURE, comb='avg', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.T = T
        self.comb = comb
        self.add_state("MR", default=torch.zeros((1, T)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros((1, T)), dist_reduce_fx="sum")

    def update(self, y_pred, y_true, heading=0):
        MR = compute_missrate(y_pred, y_true, heading, self.T, self.comb)
        self.MR += torch.Tensor(MR).sum(axis=0)
        self.count += torch.Tensor(MR).shape[0]

    def compute(self):
        return self.MR / self.count