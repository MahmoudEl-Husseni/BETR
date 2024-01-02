from config import *

import numpy as np
from itertools import combinations
from scipy.interpolate import UnivariateSpline

import torch
from torch_geometric.data import Batch, Data


def normalize(vector, point):
  return vector - point


def interpolate_x(timesteps, z):
    n_ts = (timesteps.max() - timesteps.min() + 1)
    x = timesteps
    yx = z
    if n_ts > len(x) and len(x) > 3:
      interp_func_X = UnivariateSpline(x, yx)

      yx_ = []
      it = 0
      for i in range(x.min(), x.max()+1):
        if i not in x:
          yx_.append(interp_func_X(i))
        else:
          yx_.append(yx[it])
          it+=1
    else :
      return yx

    return np.array(yx_)
  
def get_interpolated_xy(timesteps, x_coord, y_coord):
  n_ts = (timesteps.max() - timesteps.min() + 1)
  x = timesteps
  yx = x_coord
  yy = y_coord
  try: 
    if n_ts > len(x) and len(x) > 3:
      interp_func_X = UnivariateSpline(x, yx)
      interp_func_Y = UnivariateSpline(x, yy)

      yx_ = []
      yy_ = []
      it = 0
      for i in range(x.min(), x.max()+1):
        if i not in x:
          yx_.append(interp_func_X(i))
          yy_.append(interp_func_Y(i))
        else:
          yx_.append(yx[it])
          yy_.append(yy[it])
          it+=1
    else :
      return yx, yy
  except: 
    return np.array(x_coord), np.array(y_coord)
    
  return np.array(yx_), np.array(yy_)


def Angle_Distance_from_agent(df, loader):
  track_id = loader.focal_track_id
  positions = df.loc[df['track_id']==track_id, ['position_x', 'position_y']].values
  t_ids = df.loc[df['track_id']!=track_id, 'track_id'].unique()
  df['displacement_from_agent'] = np.zeros(len(df))
  df['angle_to_agent'] = np.zeros(len(df))

  for id in t_ids:
    dd = df.loc[df['track_id']==id]
    t = dd['timestep'].values
    agent_p = positions[t - t.min()]
    diff = agent_p - dd[['position_x', 'position_y']].values

    angles = np.arctan(diff[:, 1] / diff[:, 0])
    dd['angle_to_agent'] = angles

    disp = np.linalg.norm(diff, axis=1)
    dd['displacement_from_agent'] = disp

    df[df['track_id']==id] = dd.values

def n_candidates(df, loader, distance):
  Angle_Distance_from_agent(df, loader)
  df['is_candidate'] = df['displacement_from_agent'].apply(lambda x : x <= distance)

  return df.groupby('timestep')['is_candidate'].sum().values


def calc_direction(xyz):
  direction = []
  lastx, lasty, lastz = xyz[0]
  for d in xyz[1:]:
    x, y, z = d
    dir = (y - lasty) / (x - lastx)
    lastx, lasty, lastz = [x, y, z]
    direction.append(dir)

  direction.append(dir)
  return np.array(direction)


def get_avg_vectors(data, col, n_frames_per_vector, n_past=N_PAST):
  start_x = data[col][:N_PAST][:-n_frames_per_vector:n_frames_per_vector]
  end_x = data[col][:N_PAST][n_frames_per_vector::n_frames_per_vector]
  x_avg = (start_x + end_x) / 2.0
  return x_avg

def fc_graph(num_nodes): 
  edges = np.array(list(combinations(range(num_nodes), 2)))
  edges2 = edges[:, ::-1]

  edge_index = torch.tensor(np.vstack([edges, edges2]), dtype=torch.long).t().contiguous()
  
  return edge_index
  
def progress_bar(i, train_set_len, train_bs, length=70):
  """
  Displays a progress bar indicating the completion of a training step.

  Args:
    i (int): The current training step.
    length (int, optional): The length of the progress bar in characters. Defaults to 70.

  Returns:
    A string representing the progress bar.
  """

  train_steps = (train_set_len / train_bs).__ceil__()

  progress = (i+1)/train_steps
  eq = '='
  progress_bar = f"{red}{'progress:'}{res} {(f'{(progress*100):.2f}'+' %').ljust(7)} [{f'{eq*int(i*length/train_steps)}>'.ljust(length)}]"
  return progress_bar


def create_agent_graph_data(batch_data, num_nodes):
  '''
  Create bs graphs:-
  input:- list of records
  '''

  batch_data_ls = []
  batches = []

  for i, raw_data in enumerate(batch_data):
    data = Data()
    data.x = raw_data

    edge_index = fc_graph(num_nodes)

    data.edge_index = edge_index

    batch_data_ls.append(data)
    batches += [*[i]*num_nodes]

  base_data = Batch.from_data_list(batch_data_ls)
  base_data.batch = torch.tensor(batches)

  return base_data


def create_obj_graph(obj, num_nodes):
  graph_list = []
  batches = []
  for i, raw_data in enumerate(obj):

    data = Data()
    data.x = raw_data

    edge_index = fc_graph(num_nodes)

    data.edge_index = edge_index

    graph_list.append(data)
    batches += [*[i]*num_nodes]

  base_data = Batch.from_data_list(graph_list)
  base_data.batch = torch.Tensor(batches).to(int)

  return base_data