from config import *
from utils.geometry import calc_direction, n_candidates, Angle_Distance_from_agent

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


def argparser(): 
    parser = argparse.ArgumentParser(description="Process data and save to a specified directory")

    # Add arguments
    parser.add_argument("--data-dir", required=True, help="Directory containing input data")
    parser.add_argument("--save-dir", required=True, help="Directory to save processed data")
    parser.add_argument("--type", required=True, help="Type of data to process (train, val, test)")

    # Parse command-line arguments
    args = parser.parse_args()

    return args

def extract_agent_features(loader):
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



def extract_obj_features(df, loader):
  
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
    obj_df = df[df['track_id']==t_id]

    # XY_past
    t = obj_df['timestep'].values
    yx = obj_df['position_x'].values
    yy = obj_df['position_y'].values

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
    ANGLE = np.vstack((angle, angle))
    HEADING = np.vstack((HEADING, heading))
    VELOCITY = np.vstack((VELOCITY, np.hstack((vx, vy))))

    mask_tovectors.append(slice(end, end+len(yx_)))
    end += len(XYTs)

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


def extract_lane_features(avm, center, radius):
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
