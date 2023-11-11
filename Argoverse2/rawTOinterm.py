from config import *
from utils import normalize, get_interpolated_xy


import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

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


  vec = agent_df[agent_df['timestep']<N_PAST][['position_x', 'position_y']].values
  center = cur_df.loc[cur_df['track_id']==track_id_, ['position_x', 'position_y']].values.reshape(-1)

  XY_past = normalize(vec, center)

  agent_type = np.zeros(N_PAST)
  timesteps = agent_df['timestep']
  track_id = np.ones(N_PAST) * int(agent_df.iloc[0]['track_id'])

  fl_points = agent_df.sort_values(by='timestep').iloc[[0, N_PAST]].loc[:, ['position_x', 'position_y']].values
  radius = np.linalg.norm(fl_points[1]-fl_points[0])

  cur_df['displacement'] = cur_df.apply(lambda x : np.linalg.norm((x['position_x'] - center[0], x['position_y'] - center[1])), axis=1)
  cur_df['is_candidate'] = (cur_df['displacement'] <= (RADIUS_OFFSET*radius)) & (cur_df['track_id']!=track_id_)
  candidate_ids = cur_df.loc[cur_df['is_candidate'], 'track_id'].values
  
  gt = agent_df.loc[agent_df['timestep']>=N_PAST, ['position_x', 'position_y']].values
  gt_normalized = normalize(gt, center)

  data = {
      'XY_past' : XY_past, 
      'agent_type' : agent_type, 
      'timesteps' : timesteps, 
      'track_id' : track_id, 
      'candidate_ids' : candidate_ids, 
      'gt' : gt, 
      'gt_normalized' : gt_normalized 
  }

  return data 


def extract_obj_features(df, focal_track_id):

  norm_vec = df.loc[(df['track_id']==focal_track_id) & (df['timestep']==N_PAST), ['position_x', 'position_y']].values.reshape(-1)
  obj_track_ids = df.loc[df['track_id']!=focal_track_id, 'track_id'].unique()
  
  XYTs = np.empty((0, 5))
  object_types = np.empty((0, 1))
  mask_tovectors = []
  end = 0

  for t_id in obj_track_ids: 
    obj_df = df[df['track_id']==t_id]
    t = obj_df['timestep']
    yx = obj_df['position_x']
    yy = obj_df['position_y']

    yx_, yy_ = get_interpolated_xy(t, yx, yy)
    yx_norm = normalize(yx_, norm_vec[0]).reshape(-1, 1)
    yy_norm = normalize(yy_, norm_vec[1]).reshape(-1, 1)
    
    timesteps = np.arange(t.min(), t.max()+1)
    object_type = obj_df['object_type'].iloc[0]

    XYTs = np.vstack((XYTs, np.hstack((yx_.reshape(-1, 1), yy_.reshape(-1, 1), yx_norm, yy_norm, timesteps.reshape(-1, 1)))))
    mask_tovectors.append(slice(end, end+len(yx_)))  
    end += len(XYTs)

  data = {
      'XYTs' : XYTs, 
      'object_type' : object_type, 
      'mask_tovectors' : mask_tovectors
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
        agent_data = extract_agent_features(loader)
        obj_data = extract_obj_features(loader.df, loader.focal_track_id)

        np.savez(os.path.join(SAVE_DIR, scene.split('/')[-1] + "_agent"), **agent_data)
        np.savez(os.path.join(SAVE_DIR, scene.split('/')[-1] + "_obj"), **obj_data)

if __name__ == "__main__":
    main()
