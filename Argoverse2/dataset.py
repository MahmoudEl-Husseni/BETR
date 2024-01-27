from config import * 

import os
import numpy as np
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

class Agentset(Dataset):
  '''
  Agentset
  Dataset for agent vectors, ground truth
  '''
  def __init__(self, dir, normalize=False):
    self.n_features = 8
    self.n_vec = 59
    self.normalize = normalize


    self.main_dir = dir

    self.sub_dir = 'agents'
    self.suff = '_agent_vector.npy'
    
    self.gt_sub_dir = 'gt'
    self.gt_suff = '_gt.npy'

    self.prefs = os.listdir(os.path.join(self.main_dir, self.sub_dir))
    self.prefs = [i.split('_')[0] for i in self.prefs]
    

    self.agent_indices = [0, 1, 2, 3, 5, 7, 8, 9] 


    if self.normalize:
      self.means = np.load(AGENT_MEANS)
      self.stds = np.load(AGENT_STDS)

      self.gt_means = np.load(GT_MEANS)
      self.gt_stds = np.load(GT_STDS)


  def __len__(self):
    return len(self.prefs)

  def __getitem__(self, prefs):

    # Load data
    X = np.empty((0, self.n_vec, self.n_features))
    GT = np.empty((0, 50, 2))

    for pref in prefs:
        file_name = pref + self.suff

        file_path = os.path.join(self.main_dir, self.sub_dir, file_name)
        data = np.load(file_path) # 59 * 8
        x = data[:, self.agent_indices].reshape(1, self.n_vec, self.n_features)
        
        if self.normalize: 
          x = (x - self.means) / self.stds

        X = np.vstack([X, x])

        # Load GT
        gt_name = pref + self.gt_suff
        gt_path = os.path.join(self.main_dir, self.gt_sub_dir, gt_name)
        gt = np.load(gt_path).reshape(1, 50, 2)

        if self.normalize: 
          gt = (gt - self.gt_means) / self.gt_stds

        GT = np.vstack([GT, gt])

    return torch.Tensor(X), torch.Tensor(GT)  



class Objectset(Dataset):
  '''
  Objectset
  Dataset for object vectors
  for each pref (scene) has N objects returns N * 60 * 11
  '''
  def __init__(self, dir, exp_name, normalize=False, pad_mx=OBJ_PAD_LEN):
    
    self.exp_name = exp_name
    self.n_features = 11
    self.n_vec = 60
    self.normalize = normalize
    self.pad_mx = pad_mx

    self.main_dir = dir
    self.sub_dir = 'obj'
    self.suff = '_obj_vector.npz'
    self.prefs = os.listdir(os.path.join(self.main_dir, self.sub_dir))


    if self.normalize:
      self.means = np.load(OBJ_MEANS)
      self.stds = np.load(OBJ_STDS)
      self.means = np.concatenate([self.means[:4], [29], self.means[4:]])
      self.stds = np.concatenate([self.stds[:4], [15.3598], self.stds[4:]])

    if self.exp_name=='Argo-pad': 
      self.v_pad = [1000, 1000, 1000, 1000, -1000, 1000, 0, 0.0, 0.0, 0.0, 0]
      self.pade_vectors = np.array([self.v_pad for i in range(self.n_vec)])


  def __len__(self):
    return len(self.prefs)

  def __getitem__(self, prefs):
    
    x = np.empty((0, self.n_features))
    
    for pref in prefs:
      file_name = pref + self.suff
      file_path = os.path.join(self.main_dir, self.sub_dir, file_name)
      data = np.load(file_path, allow_pickle=True)
      
      data = data['vec'][:, :-1]


      if self.normalize:
        data = (data - self.means) / self.stds

      x = np.vstack((x, data))

      if self.exp_name=='Argo-pad': 
        reshaped = x.reshape(-1, 60, 11)
        disp = reshaped[:, :, 5].mean(axis=1).reshape(-1)
        
        disp = [float(x) for x in disp]
        
        disp_mp = dict(zip(disp, range(len(disp))))
        disp = sorted(disp_mp)[:self.pad_mx]

        indices = list(map(lambda x : disp_mp[x], disp))
        reshaped = reshaped[indices]

        n_pad = self.pad_mx - reshaped.shape[0]
        if n_pad>0: 
          pade_mat = np.array([self.pade_vectors for i in range(n_pad)])
          reshaped = np.vstack([reshaped, pade_mat])

        x = reshaped.reshape(-1, 11)
    return x



class Laneset(Dataset):
  '''
  Laneset
  Dataset for lane vectors
  for each pref (scene) has N lanes returns N * 35 * 9
  '''
  def __init__(self, dir, exp_name, normalize=False, pad_mx=LANE_PAD_LEN):

    self.exp_name = exp_name
    self.normalize = normalize
    self.n_features = 9
    self.n_vec = 35
    self.pad_mx = pad_mx

    self.main_dir = dir
    self.sub_dir = 'lanes'
    self.suff = '_lane_vector.npz'
    self.prefs = os.listdir(os.path.join(self.main_dir, self.sub_dir))
    self.lane_indices = [*range(9)] # [N * 9]

    if self.normalize: 
      self.means = np.load(LANE_MEANS)
      self.stds = np.load(LANE_STDS)

    if self.exp_name=='Argo-pad':
      self.v_pad = [1000, 1000, 1000, 1000, 1000, 1000, 0, 0, 0]
      self.pade_vectors = np.array([self.v_pad for i in range(self.n_vec)])


  def __len__(self):
    return len(self.prefs)

  def __getitem__(self, prefs):
    
    x = np.empty((0, self.n_features))
    
    for pref in prefs: 
      file_name = pref + self.suff
      file_path = os.path.join(self.main_dir, self.sub_dir, file_name)
      data = np.load(file_path, allow_pickle=True)
      data = data['vec'][:, :-1]
      
      if self.normalize: 
        data = (data - self.means) / self.stds

      x = np.vstack((x, data))

      if self.exp_name=='Argo-pad':
        reshaped = x.reshape(-1, 35, 9)
        disp = np.linalg.norm(reshaped[:, :, :2], axis=-1).mean(axis=-1)
        disp_mp = dict(zip(disp, range(len(disp))))
        disp = sorted(disp_mp)[:self.pad_mx]

        indices = list(map(lambda x : disp_mp[x], disp))
        reshaped = reshaped[indices]

        n_pad = self.pad_mx - reshaped.shape[0]
        if n_pad>0:
          pade_mat = np.array([self.pade_vectors for i in range(n_pad)])
          reshaped = np.vstack([reshaped, pade_mat])

        x = reshaped.reshape(-1, 9)

    return x
  
  


class Vectorset(Dataset):
  '''
  Vectorset
  Dataset for agent, object, lane vectors, ground truth
  '''
  def __init__(self, dir, exp_name, normalize=False):
    self.main_dir = dir
    self.normalize = normalize
    self.exp_name = exp_name

    self.agent_set = Agentset(self.main_dir, self.normalize)
    self.obj_set = Objectset(self.main_dir, self.exp_name, self.normalize)
    self.lane_set = Laneset(self.main_dir, self.exp_name, self.normalize)
    
    self.prefs = self.agent_set.prefs


  def __len__(self):
    return len(self.prefs)

  def __getitem__(self, idx):
    if isinstance(idx, int): 
      prefs = [self.prefs[idx]]
    else : 
      prefs = self.prefs[idx]

    agent_vectors, gt = self.agent_set[prefs]
    obj_vectors = self.obj_set[prefs]
    lane_vectors = self.lane_set[prefs]

    return agent_vectors, obj_vectors, lane_vectors, gt





def custom_collate(batch: List[Tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[float]]:
    """
    Custom collate function for PyTorch DataLoader.

    Args:
        batch (List[Tuple[torch.Tensor, List[List[float]], List[List[float]], torch.Tensor]]): 
            A batch of samples, where each sample is a tuple containing:
            - agent (torch.Tensor): Tensor representing agent features.
            - obj (List[List[float]]): List of object features.
            - lane (List[List[float]]): List of lane features.
            - gt (torch.Tensor): Tensor representing ground truth features.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float], List[float]]:
            - agent (torch.Tensor): Concatenated tensor of agent features.
            - obj (torch.Tensor): Concatenated tensor of object features.
            - lane (torch.Tensor): Concatenated tensor of lane features.
            - gt (torch.Tensor): Concatenated tensor of ground truth features.
            - n_objs (List[float]): List of cumulative counts of objects in the batch.
            - n_lanes (List[float]): List of cumulative counts of lanes in the batch.
    """
    agent_feat = batch[0][0].shape[-1]
    # obj_feat = len(batch[0][1][0])
    # lane_feat = len(batch[0][2][0])
    obj_feat = 11
    lane_feat = 9
    gt_feat = batch[0][3].shape[-1]

    agent = torch.empty((0, 59, agent_feat))
    obj = torch.empty((0, 60, obj_feat))
    lane = torch.empty((0, 35, lane_feat))
    gt = torch.empty((0, batch[0][3].shape[-2], gt_feat))
    n_objs = [0]
    n_lanes = [0]
    obj_sm = 0
    lane_sm = 0
    for b in batch:
        agent = torch.cat([agent, b[0]])
        
        obj = torch.cat([obj, torch.Tensor(b[1].astype('float')).view(-1, 60, obj_feat)])
        lane = torch.cat([lane, torch.Tensor(b[2]).view(-1, 35, lane_feat)])
        gt = torch.cat([gt, torch.Tensor(b[3])])
        obj_sm += b[1].shape[0] / 60
        lane_sm += b[2].shape[0] / 35
        n_objs.append(obj_sm)
        n_lanes.append(lane_sm)

    return agent.to(DEVICE), obj.to(DEVICE), lane.to(DEVICE), gt.to(DEVICE), n_objs, n_lanes


class SingleScene(Vectorset):
  def __init__(self, dir, exp_name, normalize=False): 
    super().__init__(dir, exp_name, normalize)

  def __getitem__(self, scene_name): 
    if isinstance(scene_name, int): 
      prefs = [scene_name]
    else : 
      prefs = scene_name

    agent_vectors, gt = self.agent_set[prefs]
    obj_vectors = self.obj_set[prefs]
    lane_vectors = self.lane_set[prefs]

    return agent_vectors, obj_vectors, lane_vectors, gt