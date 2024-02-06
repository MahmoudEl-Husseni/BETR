import sys
sys.path.append('VectorNet/Argoverse2')

from config import * 
from dataset import * 
from Vectornet import * 
from metrics.ade import *
from metrics.fde import *
from metrics.mr import *

import argparse
from tqdm import tqdm
import time



def normalize(y, mean, std):
  return (y-mean) / std


def eval_dataloader(model, dataloader, metrics): 

    mean = torch.Tensor(np.load('VectorNet/Argoverse2/stats/gt/gt_means.npy'))
    std = torch.Tensor(np.load('VectorNet/Argoverse2/stats/gt/gt_stds.npy'))
    model.eval()
    Ts = []
    for batch in tqdm(iter(dataloader)):
    
        start_time = time.time()
        out = model(batch)
        t = time.time() - start_time
        Ts.append(t)
        pred, confidences = out[:, :-N_TRAJ], out[:, -N_TRAJ:]
        pred = pred.view(-1, N_TRAJ, N_FUTURE, 2)
        target = batch[3]
        if EXPERIMENT_NAME=='Argo-1':
          target = normalize(target, mean, std)
          pred = normalize(pred, mean, std)


        for metric in metrics : 
            metric.update(pred, target)
        

    return [metric.compute() for metric in metrics], Ts


def argparser(): 
    parser = argparse.ArgumentParser(description='VectorNet')
    parser.add_argument('--experiment_name', type=str, default='Argo-1')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--best_models', type=str, default='')
    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--typ', type=str, default='')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = argparser()
    EXPERIMENT_NAME = args.experiment_name
    DATA_PATH = args.data_path

    if EXPERIMENT_NAME=='Argo-1' : 
        dataset = Vectorset(DATA_PATH, EXPERIMENT_NAME, normalize=False)
    else :
        dataset = Vectorset(DATA_PATH, EXPERIMENT_NAME, normalize=True)

    model = VectorNet(EXPERIMENT_NAME)
    model.exp_name = EXPERIMENT_NAME
    model_ckpt_path = args.best_models + f'/{EXPERIMENT_NAME.lower()}-best_model.pth'

    checkpoint = torch.load(model_ckpt_path, map_location='cpu')


    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)

    valloader = DataLoader(dataset[:10000], batch_size=VAL_BS, shuffle=True, collate_fn=custom_collate)

    mde_metric = MinMDE()
    fde_metric = MinFDE()
    mr_metric = MissRate()

    # metrics = [mde_metric, fde_metric, mr_metric]

    [mde, fde, mr], Ts = eval_dataloader(model, valloader, metrics)
    np.save(f'{EXPERIMENT_NAE}_T.npy', np.array(Ts))
    mr = mr.squeeze()
    
    with open(args.results_path, 'a') as f:
      f.write(f'{EXPERIMENT_NAME}_{args.typ},{fde},{mde},' + ','.join(map(lambda x : str(float(x)), mr)) + '\n')