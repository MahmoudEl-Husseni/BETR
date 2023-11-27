import os 
import numpy as np

def pad_obj_vectors(obj_vec, ts, n_vec=60, inf = 1000):
  diff_s = int(ts[0])
  diff_e = int(n_vec - ts[-1])

  obj_vec_tensor = np.array(obj_vec)
  # [xs, ys, xe, ye, ts_vg, dis, ang, vx, vy, heading, obj_type, Pid]

  pad_s = [obj_vec[0].tolist()]
  pad_e = [obj_vec[-1].tolist()]
  pad_s = np.array(pad_s * diff_s)
  pad_e = np.array(pad_e * diff_e)
  if len(pad_s)==0:
    pad_s = np.empty((0, obj_vec.shape[1]))

  if len(pad_e)==0:
    pad_e = np.empty((0, obj_vec.shape[1]))


  obj_vec = np.vstack([pad_s, obj_vec_tensor, pad_e])
  return obj_vec


def pad_lane_vectors(lane_vec, n_vec=35, inf = 1000):
  diff = n_vec - len(lane_vec)

  lane_vec_tensor = np.array(lane_vec)
  # [xs, ys, zs, xe, ye, ze, is_inter, dir, type, line_id]

  pad = [lane_vec_tensor[0].tolist()]
  pad = np.array(pad * diff)
  if len(pad)>0:
    lane_vec = np.vstack([pad, lane_vec_tensor])
  return lane_vec[:n_vec]


def save_agent(agent_vector, pref, main_dir, agent_dir):
  '''
  Save Agent Vector: [50, 8]
  '''
  file_name = f"{pref}_agent_vector"
  file_path = os.path.join(main_dir, agent_dir, file_name)

  np.save(file_path, agent_vector)


def save_objects(vectors, pref, main_dir, obj_dir):
  '''
  Save Object Vectors: [L * 60, 12]
  L * 60 object vectors, each object vector has 12 features
  '''
  for i in range(0, len(vectors), 60):
    file_name = f"{pref}_{str(i//60).rjust(4, '0')}"
    file_path = os.path.join(main_dir, obj_dir, file_name)
    np.save(file_path, vectors[i:i+60])


def save_lanes(vectors, pref, main_dir, lane_dir):
  '''
  Save Lane Vectors: [L * 35, 10]
  L * 35 lane vectors, each lane vector has 10 features
  '''
  for i in range(0, len(vectors), 35):
    file_name = f"{pref}_{str(i//35).rjust(4, '0')}"
    file_path = os.path.join(main_dir, lane_dir, file_name)
    np.save(file_path, vectors[i:i+35])

def save_gt(gt, pref, main_dir, gt_dir):
  '''
  save ground truth [50, 2]
  '''
  file_name = f"{pref}_gt"
  file_path = os.path.join(main_dir, gt_dir, file_name)

  np.save(file_path, gt)