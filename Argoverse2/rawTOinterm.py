from config import *
from utils.geometry import calc_direction, n_candidates, Angle_Distance_from_agent, get_avg_vectors

from Argoverse2.utils.visualize import normalize, get_interpolated_xy


import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


sys.path.append("/content/av2-api/src")

from av2.datasets.motion_forecasting import scenario_serialization as ss
from av2.map.map_api import ArgoverseStaticMap
# =================================================================

def argparser(): 
    parser = argparse.ArgumentParser(description="Process data and save to a specified directory")

    # Add arguments
    parser.add_argument("--data-dir", required=True, help="Directory containing input data")
    parser.add_argument("--save-dir", required=True, help="Directory to save processed data")
    parser.add_argument("--type", required=True, help="Type of data to process (train, val, test)")

    # Parse command-line arguments
    args = parser.parse_args()

    return args


def extract_agent_features(loader): # Time Complexity -> O(1) // discarding numpy orperations & Loading from Disk
  df = ss._convert_tracks_to_tabular_format(loader.tracks)
  track_id_ = loader.focal_track_id
  agent_df = df[df['track_id']==track_id_]
  cur_df = df[df['timestep']==N_PAST-1]

  # Past XY
  vec = agent_df[agent_df['timestep']<N_PAST][['position_x', 'position_y']].values
  center = cur_df.loc[cur_df['track_id']==track_id_, ['position_x', 'position_y']].values.reshape(-1)

  XY_past = normalize(vec, center)

  # Timestep
  timesteps = agent_df['timestep'].values

  # Vx, Vy
  Vx = agent_df['velocity_x'].values
  Vy = agent_df['velocity_y'].values

  # Heading
  heading = agent_df['heading'].values

  # Candidates Denisity
  fl_points = agent_df.sort_values(by='timestep').iloc[[0, N_PAST]].loc[:, ['position_x', 'position_y']].values
  radius = np.linalg.norm(fl_points[1]-fl_points[0])
  CD = n_candidates(df, loader, radius * RADIUS_OFFSET)

  # Agent type
  agent_type = np.zeros(N_PAST)

  # Track ID
  track_id = np.ones(N_PAST) * int(agent_df.iloc[0]['track_id'])


  # cur_df['displacement'] = cur_df.apply(lambda x : np.linalg.norm((x['position_x'] - center[0], x['position_y'] - center[1])), axis=1)
  # cur_df['is_candidate'] = (cur_df['displacement'] <= (RADIUS_OFFSET*radius)) & (cur_df['track_id']!=track_id_)
  # candidate_ids = cur_df.loc[cur_df['is_candidate'], 'track_id'].values

  # Ground Truth Trajectory
  gt = agent_df.loc[agent_df['timestep']>=N_PAST, ['position_x', 'position_y']].values

  # Normalized GT
  gt_normalized = normalize(gt, center)

  data = {
      'XY_past' : XY_past,
      'timesteps' : timesteps,
      'vx' : Vx,
      'vy' : Vy,
      'heading' : heading,
      'candidate_density' : CD,
      'agent_type' : agent_type,
      'track_id' : track_id,
      # 'candidate_ids' : candidate_ids,
      'gt' : gt,
      'gt_normalized' : gt_normalized
  }
  return data, center, radius


def extract_obj_features(df, loader): # Time Complexity -> O(n) (n: Number of objects)

  focal_track_id = loader.focal_track_id

  norm_vec = df.loc[(df['track_id']==focal_track_id) & (df['timestep']==N_PAST), ['position_x', 'position_y']].values.reshape(-1)
  obj_track_ids = df.loc[df['track_id']!=focal_track_id, 'track_id'].unique()

  XYTs = np.empty((0, 5))
  DIST = np.empty((0, 1))
  ANGLE = np.empty((0, 1))
  HEADING = np.empty((0, 1))
  VELOCITY = np.empty((0, 2))
  OBJECT_TYPE = []

  mask_tovectors = []
  end = 0
  p_id = 0

  for t_id in obj_track_ids:

    Angle_Distance_from_agent(df, loader)
    obj_df = df[(df['track_id']==t_id) & (df['timestep']<N_PAST)]

    # XY_past
    t = obj_df['timestep'].values
    yx = obj_df['position_x'].values
    yy = obj_df['position_y'].values
    if len(t) < 1: 
      continue
    yx_, yy_ = get_interpolated_xy(t, yx, yy)
    yx_norm = normalize(yx_, norm_vec[0]).reshape(-1, 1)
    yy_norm = normalize(yy_, norm_vec[1]).reshape(-1, 1)

    # Distance From Agent
    distance = obj_df['displacement_from_agent'].values.reshape(-1, 1)
    angle = obj_df['angle_to_agent'].values.reshape(-1, 1)

    # Heading
    heading = obj_df['heading'].values.reshape(-1, 1)

    # velocity
    vx = obj_df['velocity_x'].values.reshape(-1, 1)
    vy = obj_df['velocity_y'].values.reshape(-1, 1)

    # Polyline id
    polyline_id = p_id

    # timestep
    timesteps = np.arange(t.min(), t.max()+1)

    # Object type
    OBJECT_TYPE.append([obj_df['object_type'].iloc[0]] * len(timesteps))

    XYTs = np.vstack((XYTs, np.hstack((yx_.reshape(-1, 1), yy_.reshape(-1, 1), yx_norm, yy_norm, timesteps.reshape(-1, 1)))))
    DIST = np.vstack((DIST, distance))
    ANGLE = np.vstack((ANGLE, angle))
    HEADING = np.vstack((HEADING, heading))
    VELOCITY = np.vstack((VELOCITY, np.hstack((vx, vy))))

    mask_tovectors.append(slice(end, len(XYTs)))
    end = len(XYTs)

    p_id += 1
  data = {
      'XYTs' : XYTs,
      'DIST' : DIST,
      'ANGLE' : ANGLE,
      'HEADING' : HEADING,
      'VELOCITY' : VELOCITY,
      'object_type' : OBJECT_TYPE,
      'mask_tovectors' : mask_tovectors
  }
  return data


def extract_lane_features(avm, center, radius): # Time Complexity -> O(n) (n: No. Polylines)
  polylines = avm.get_nearby_lane_segments(center, radius*RADIUS_OFFSET)
  XYZ = []
  IS_INTER = []
  TYP = []
  ID = []
  DIRECTION = []

  for poly in polylines:
    xyz = poly.polygon_boundary
    is_inter = poly.is_intersection
    typ = poly.lane_type
    id = poly.id
    dir = calc_direction(xyz)


    XYZ.append(xyz)
    IS_INTER.append(is_inter)
    TYP.append(typ)
    ID.append(id)
    DIRECTION.append(dir)

  data = {
      "XYZ" : XYZ,
      "IS_INTER" : IS_INTER,
      "TYP" : TYP,
      "ID" : ID,
      "DIRECTION" : DIRECTION
  }

  return data


# ========================= Vectorize Data =========================
def vectorize_agent(agent_data, dt=TRAJ_DT, sample_rate=ARGO_SAMPLE_RATE):

  n_frames_per_vector = int(sample_rate * dt)
 
  Xs = agent_data['XY_past'][:-n_frames_per_vector:n_frames_per_vector, 0].reshape(-1, 1)
  Ys = agent_data['XY_past'][:-n_frames_per_vector:n_frames_per_vector, 1].reshape(-1, 1)

  Xe = agent_data['XY_past'][n_frames_per_vector::n_frames_per_vector, 0].reshape(-1, 1)
  Ye = agent_data['XY_past'][n_frames_per_vector::n_frames_per_vector, 1].reshape(-1, 1)

  ts_avg = get_avg_vectors(agent_data, 'timesteps', n_frames_per_vector, N_PAST).reshape(-1, 1)
  CD_avg = get_avg_vectors(agent_data, 'candidate_density', n_frames_per_vector, N_PAST).reshape(-1, 1)
  vx_avg = get_avg_vectors(agent_data, 'vx', n_frames_per_vector, N_PAST).reshape(-1, 1)
  vy_avg = get_avg_vectors(agent_data, 'vy', n_frames_per_vector, N_PAST).reshape(-1, 1)
  heading = get_avg_vectors(agent_data, 'heading', n_frames_per_vector, N_PAST).reshape(-1, 1)
  zeros = np.zeros((len(heading), 1))

  vectors = np.hstack((Xs, Ys, Xe, Ye, ts_avg, CD_avg, zeros, vx_avg, vy_avg, heading, zeros))
  return vectors


def vectorize_obj(obj_data, dt=TRAJ_DT, sample_rate=ARGO_SAMPLE_RATE):
  vectors = np.empty((0, 11))
  n_frames_per_vector = int(sample_rate * dt)
  mask = []

  for i, _mask in enumerate(obj_data['mask_tovectors'], start=1): 
    Xs = obj_data['XYTs'][_mask][:-n_frames_per_vector:n_frames_per_vector, 2].reshape(-1, 1)
    Ys = obj_data['XYTs'][_mask][:-n_frames_per_vector:n_frames_per_vector, 3].reshape(-1, 1)

    Xe = obj_data['XYTs'][_mask][n_frames_per_vector::n_frames_per_vector, 2].reshape(-1, 1)
    Ye = obj_data['XYTs'][_mask][n_frames_per_vector::n_frames_per_vector, 3].reshape(-1, 1)



    ts_start = obj_data['XYTs'][_mask][n_frames_per_vector::n_frames_per_vector, 4].reshape(-1, 1)
    ts_end = obj_data['XYTs'][_mask][n_frames_per_vector::n_frames_per_vector, 4].reshape(-1, 1)
    ts_avg = (ts_start + ts_end) / 2.0

    dist_start = obj_data['DIST'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    dist_end = obj_data['DIST'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    dist_avg = (dist_start + dist_end) / 2.0

    angle_start = obj_data['ANGLE'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    angle_end = obj_data['ANGLE'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    angle_avg = (angle_start + angle_end) / 2.0

    vx_start = obj_data['VELOCITY'][_mask, 0][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    vx_end = obj_data['VELOCITY'][_mask, 0][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    vx_avg = (vx_start + vx_end) / 2.0

    vy_start = obj_data['VELOCITY'][_mask, 1][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    vy_end = obj_data['VELOCITY'][_mask, 1][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    vy_avg = (vy_start + vy_end) / 2.0


    heading_start = obj_data['HEADING'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    heading_end = obj_data['HEADING'][_mask][n_frames_per_vector::n_frames_per_vector].reshape(-1, 1)
    heading_avg = (heading_start + heading_end) / 2.0

    p_id = (np.ones(len(Xs)) * i).reshape(-1, 1)

    _vectors = np.hstack([Xs, Ys, Xe, Ye, ts_avg, dist_avg, angle_avg, vx_avg, vy_avg, heading_avg, p_id])
    vectors = np.vstack([vectors, _vectors])
    mask.append(slice(len(vectors) - len(_vectors), len(vectors)))
  
  return vectors, mask

# ============================== main ==============================
def main():
    args = argparser()
    DATA_DIR = args.data_dir
    SAVE_DIR = os.path.join(args.save_dir, args.type)
    SCENE_DIRS = [os.path.join(DATA_DIR, "train")]

    # Create save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for scene in tqdm(SCENE_DIRS): 
        file_name = "scenario_" + scene.split('/')[-1] + ".parquet"
        loader = ss.load_argoverse_scenario_parquet(file_name)
        
        log_map_dirpath = Path(scene)
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath=log_map_dirpath, build_raster=False)


        agent_data, center, radius = extract_agent_features(loader)
        obj_data = extract_obj_features(loader.df, loader.focal_track_id)
        lane_data = extract_lane_features(avm, center, radius)
        
        np.savez(os.path.join(SAVE_DIR, scene.split('/')[-1] + "_agent"), **agent_data)
        np.savez(os.path.join(SAVE_DIR, scene.split('/')[-1] + "_obj"), **obj_data)

if __name__ == "__main__":
    main()
