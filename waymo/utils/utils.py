import os
import time
import random
import numpy as np

def generate_pref():

  # Generate a random number of 10 digits
  random_number = random.randint(1000000, 99999999)

  # Get the current time in seconds
  current_time = int(time.time() * 1000)

  # Attach the random number to the current time
  pref = str(current_time) + '_' + str(random_number)

  return pref



def get_lane_splitter(lane_id):
  unique_lane_ids = []

  for id in lane_id:
    if id in unique_lane_ids:
      continue

    unique_lane_ids.append(int(id))


  sm = []

  for id in unique_lane_ids:
    if id < 0 :
      continue
    sm.append((lane_id == id).sum())

  sm = [0] + sm

  for i in range (1, len(sm)) :
    sm[i] = sm[i-1] + sm[i]

  return sm




def agent_agent_disp(XY):
  x = XY[:, 0, 0].reshape(-1, 1)
  center = XY[:, -1, :]
  xy_norm = np.zeros_like(XY)
  n_agents = XY.shape[0]
  n_ts = XY.shape[1]


  DISP = np.empty((n_agents, n_agents, 0))

  for i in range(11):
    xy_norm[:, i, :] = XY[:, i, :]

  for i in range(11):

    x_norm = xy_norm[:, i, 0].reshape(-1, 1)
    y_norm = xy_norm[:, i, 1].reshape(-1, 1)

    disp_x = (x_norm-x_norm.T)**2
    disp_y = (y_norm-y_norm.T)**2

    disp = np.expand_dims(np.sqrt(disp_x + disp_y), -1) + np.expand_dims(np.eye(n_agents) * 100000000, -1)

    DISP = np.concatenate([DISP, disp], axis=-1)

  return DISP



def save_agents(agent_vectors, dir, pref):
  file_name = pref + "_agent"
  np.save(os.path.join(dir, file_name), agent_vectors)

def save_objects(obj_vectors, dir, pref):
  file_name = pref + "_obj"
  np.save(os.path.join(dir, file_name), obj_vectors)

def save_lanes(lane_vectors, dir, pref):
  file_name = pref + "_lane"
  np.save(os.path.join(dir, file_name), lane_vectors)


def save_gt(gt_normalized, dir, pref):
  file_name = pref + "_gt"
  np.save(os.path.join(dir, file_name), gt_normalized)

