import os
import numpy as np
import tensorflow as tf
from utils.data import *
from utils.utils import *
from config import *



def process(scene_file, save_dir):
  os.makedirs(f'{save_dir}/agents', exist_ok=True)
  os.makedirs(f'{save_dir}/obj', exist_ok=True)
  os.makedirs(f'{save_dir}/lanes', exist_ok=True)
  os.makedirs(f'{save_dir}/gt', exist_ok=True)

  data = load_data(scene_file)

  for record in data:
    XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, \
            GT_XY, Future_valid, Tracks_to_predict, lane_xyz, \
            lane_valid, lane_dir, lane_id, lane_type = record


    sm = get_lane_splitter(lane_id)
    DISP = agent_agent_disp(XY)

    for i in range(len(XY)):
      ret = extract_features(i, XY, Velocity, Current_valid, Agents_val, Agent_type, YAWS, GT_XY, Future_valid,
                            Tracks_to_predict, lane_xyz, lane_valid, lane_dir, lane_id, lane_type,
                            sm, DISP)

      if ret == -1:
        continue

      _agent_data, _obj_data, _lane_data = ret

      agent_vectors = vectorize_agent(_agent_data)
      obj_vectors = vectorize_object(_obj_data)
      lane_vectors = vectorize_lanes(_lane_data)

      pref = generate_pref()
      pref = scene_file[-14:-9] + '_' + pref
      save_agents(agent_vectors, f'{save_dir}/agents', pref)
      save_objects(obj_vectors, f'{save_dir}/obj', pref)
      save_lanes(lane_vectors, f'{save_dir}/lanes', pref)
      save_gt(_agent_data['Normalized_GT'], f'{save_dir}/gt', pref)