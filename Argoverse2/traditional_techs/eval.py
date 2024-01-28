import sys
import numpy as np

sys.path.append('/main/VectorNet/Argoverse2')
sys.path.append("av2-api/src")

import warnings
warnings.simplefilter('ignore')

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization as ss
from config import *

from metrics.mr import compute_mr
from metrics.fde import compute_fde
from metrics.ade import mean_displacement_error

import torch
from tqdm import tqdm
from rawTOinterm import extract_agent_features
import argparse



from techs import SUPPORTED_TECHS  

def normalize(y, mean, std):
  return (y-mean) / std

def argparser():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--data-dir', required=True, help='Directory containing input data')
    parser.add_argument('--results-path', required=True, help='Directory to save processed data')
    parser.add_argument('--typ', required=True, help='Type of data to process (train, val, test)')
    parser.add_argument('--exp-name', required=True)

    # Parse command-line arguments
    args = parser.parse_args()

    return args


if __name__=='__main__':
  args = argparser()

  mean = torch.Tensor(np.load('/main/VectorNet/Argoverse2/stats/gt/gt_means.npy'))
  std = torch.Tensor(np.load('/main/VectorNet/Argoverse2/stats/gt/gt_stds.npy'))
  
  exp_name = args.exp_name
  if exp_name not in SUPPORTED_TECHS: 
    raise NameError(f"{exp_name} is not supported tech") 
  if exp_name=='poly-fit': 
    from techs import poly_interp
    func = poly_interp
  elif exp_name=='interp1d': 
    from techs import interp1d_ex
    func = interp1d_ex
  elif exp_name=='CubicSpline': 
    from techs import cubicspline_ex
    func = cubicspline_ex
  
  t = np.arange(N_PAST)
  t_out = np.arange(N_PAST, N_PAST+N_FUTURE)
  
  smFDE = 0.0
  smADE = 0.0
  cnt = 0

  raw_scenes_dir = args.data_dir

  raw_scenes = [os.path.join(raw_scenes_dir, i, 'scenario_'+i+'.parquet') for i in os.listdir(raw_scenes_dir)]



  for raw_ex in tqdm(raw_scenes):

    loader = ss.load_argoverse_scenario_parquet(raw_ex)
    ex_data, center, radius = extract_agent_features(loader)
    xy_past = ex_data['XY_past']
    x = xy_past[:, 0]
    y = xy_past[:, 1]
    
    gt = ex_data['gt_normalized']
    gt = torch.Tensor(gt)


    x_pred = func(t, x, t_out)
    y_pred = func(t, y, t_out)

    pred = torch.Tensor([x_pred, y_pred]).T
    
    pred = normalize(pred, mean, std)
    gt = normalize(gt, mean, std)

    # Compute fde, ade, mr
    fde = compute_fde(pred[None, None, :], gt[None, :]).item()
    
    ade = mean_displacement_error(pred[None, None, :], gt[None, :]).squeeze()
    MR = compute_mr(pred[None, :], torch.Tensor(gt[None, :]), 50)
    MR += MR.sum(axis=0)

    smFDE += fde
    smADE += ade
    cnt += 1
    

  fde = smFDE / cnt
  ade = smADE / cnt
  mr = (MR / cnt).squeeze()
  # print(f'{args.exp_name}_{args.typ},{fde},{ade},' + ','.join(map(lambda x : str(float(x)), mr)))
  with open(args.results_path, 'a') as f:
      f.write(f'{args.exp_name}_{args.typ},{fde},{ade},' + ','.join(map(lambda x : str(float(x)), mr)) + '\n')
