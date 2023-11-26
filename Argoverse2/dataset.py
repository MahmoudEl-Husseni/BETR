from config import * 

import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

class Agentset(Dataset):
  def __init__(self, dir):
    self.main_dir = dir
    self.suff = '_agent_vectors.npz'
    self.gt_suff = '_gt_normalized.npz'

    self.prefs = os.listdir(self.main_dir)
    # [xs, ys, xe, ye, ts, cd, zeros, vx, vy, heading, zeros]
    self.agent_indices = [0, 1, 2, 3, 5, 7, 8, 9] # N * 8
  def __len__(self):
    return len(self.prefs)

  def __getitem__(self, pref):
    # Load data
    file_name = pref + self.suff
    file_path = os.path.join(self.main_dir, pref, file_name)
    data = np.load(file_path)['arr_0'] # 59 * 114
    x = data[:, self.agent_indices]

    # Load GT
    gt_name = pref + self.gt_suff
    gt_path = os.path.join(self.main_dir, pref, gt_name)
    gt = np.load(gt_path)['gt_normalized']

    return torch.Tensor(x), torch.Tensor(gt)

class Objectset(Dataset):
  def __init__(self, dir):
    self.main_dir = dir
    self.suff = '_obj_vectors.npz'
    self.suff_mask = '_obj_mask.npz'
    # [xs, ys, xe, ye, ts, Dis, angle, vx, vy, heading, object_type, Pid]
    self.object_indices = [*range(10)] # [N * 10]

  def __getitem__(self, pref):
    # Load Mask
    mask_name = pref + self.suff_mask
    mask_path = os.path.join(self.main_dir, pref, mask_name)
    masks = np.load(mask_path, allow_pickle=True)['mask']

    # Load data
    file_name = pref + self.suff
    file_path = os.path.join(self.main_dir, pref, file_name)
    data = np.load(file_path)['arr_0']
    x = data[:, self.object_indices]

    vectors = []
    for mask in masks:
      vectors.append(torch.Tensor(x[mask]))

    return vectors

class Laneset(Dataset):
  def __init__(self, dir):
    self.main_dir = dir
    self.suff = '_lane_vectors.npz'
    self.suff_mask = '_lane_mask.npz'
    # [xs, ys, zs, xe, ye, ze, is_intersection, dir, typ, lineid]
    self.object_indices = [*range(9)] # [N * 9]

  def __getitem__(self, pref):
    # Load Mask
    mask_name = pref + self.suff_mask
    mask_path = os.path.join(self.main_dir, pref, mask_name)
    masks = np.load(mask_path, allow_pickle=True)['mask']

    # Load data
    file_name = pref + self.suff
    file_path = os.path.join(self.main_dir, pref, file_name)
    data = np.load(file_path)['arr_0']
    x = data[:, self.object_indices]
    x[:, -1] = [*map(lambda x : map_object_type[x.lower()], x[:, -1])]

    vectors = []
    for mask in masks:
      vectors.append(torch.Tensor(x[mask].astype('float32')))

    return vectors

class Vectorset(Dataset):
  def __init__(self, dir):
    self.main_dir = dir
    self.agent_set = Agentset(dir)
    self.obj_set = Objectset(dir)
    self.lane_set = Laneset(dir)
    self.paths = os.listdir(self.main_dir)

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    pref = self.paths[idx]

    agent_vectors, gt = self.agent_set[pref]
    obj_vectors = self.obj_set[pref]
    lane_vectors = self.lane_set[pref]

    return agent_vectors, obj_vectors, lane_vectors, gt
