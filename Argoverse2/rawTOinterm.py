from config import *
from utils import normalize


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
  cur_df['is_candidate'] = (cur_df['displacement'] <= (1.5*radius)) & (cur_df['track_id']!=track_id_)
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
        data = extract_agent_features(loader)
        np.savez(os.path.join(SAVE_DIR, scene.split('/')[-1]), **data)

if __name__ == "__main__":
    main()