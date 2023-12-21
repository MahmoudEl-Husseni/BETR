from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from loss import * 
from config import *
from utils.visualize import *
from utils.ckpt_utils import *
from Vectornet import VectorNet
from utils.geometry import progress_bar

if EXPERIMENT_NAME=='Argo-avg': 
    from dataset_argoavg import Vectorset, custom_collate
elif EXPERIMENT_NAME=='Argo-1' or EXPERIMENT_NAME=='Argo-Normalized':
    from dataset import Vectorset, custom_collate



import time
import logging

import warnings
warnings.simplefilter('ignore')


def eval_metrics(y_true, y_pred, conf, metrics:dict): 
    results = dict()
    for name, metric in metrics.items(): 
        m = metric(y_true, y_pred, conf)
        results[name] = m
    
    return results


def get_min_loss(best_logs_file):
    if os.path.exists(best_logs_file):
        with open(best_logs_file, "r") as f:
            n = 0
            for line in f.readlines():
                n += 1
            if n >1:
                best_loss = float(line.split(",")[1])
            else :
                best_loss = 1e9
    else:
        best_loss = 1e9

    return best_loss 



def train_one_batch(model, batch, optimizer, loss_func, scheduler, metrics): 

    optimizer.zero_grad()
    out = model(batch)
    y_pred, confidences = out[:, :-N_TRAJ], out[:, -N_TRAJ:]
    y_pred = y_pred.view(-1, N_TRAJ, N_FUTURE, 2)

    gt = batch[3]
    

    l = loss_func(gt, y_pred, confidences)

    l.backward()
    optimizer.step()
    scheduler.step()

    __results = eval_metrics(gt, y_pred, confidences, metrics)

    return l.item(), __results


def train_from_last_ckpt(model, 
                         train_loader,
                         val_loader, 
                         optimizer, 
                         loss_criteria, 
                         scheduler,
                         ckpt_dir, 
                         writer, 
                         metrics):
    
    # Setup
    last_ckpt_weights = os.path.join(ckpt_dir, 'last_model.pth')
    
    if os.path.exists(last_ckpt_weights): 
        # Load weights to model
        end_epoch = load_checkpoint(last_ckpt_weights, model, optimizer)

    else : 
        # Start Training from first epoch
        end_epoch = 0
    
    # Get min loss 
    best_logs_file = os.path.join(OUT_DIR, "best_logs.csv")
    best_loss = get_min_loss(best_logs_file)
    
    tb_it_tr = 0
    tb_it_v = 0


    for epoch in range(end_epoch+1, EPOCHS):
        model.train()

        print(f'{red}{"[INFO]:  "}{res}Epoch {blk}{f"#{epoch+1}/{EPOCHS}"}{res} started')

        # Training loop


        results = dict()
        for k in metrics.keys(): 
            results[k] = []

        
        for i, data in enumerate(train_loader):
        
            print("\r", end=f'{progress_bar(i, train_set_len=len(train_loader)*TRAIN_BS, train_bs=TRAIN_BS, length=75)}')
            
            
            if i % LOG_STEP==0:
                logging.info(f"Training Epoch {epoch+1}, batch {i+1} / {len(train_loader)}")

            loss, __results = train_one_batch(model, data, optimizer, loss_criteria, scheduler, metrics)
            time.sleep(5)

            # Write results on tensorboard
            for name, result in __results.items(): 
                if isinstance(result, list): 
                    result = np.mean(result)
                writer.add_scalar(f'{name}/train', result, tb_it_tr)

            writer.add_scalar('loss/train', loss, tb_it_tr)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], tb_it_tr)
            
            tb_it_tr += 1




        # write logs to csv file
        with open(os.path.join(OUT_DIR, "logs.csv"), "a") as f:
            line = f"{epoch+1}"

            for result in __results.values():
                if isinstance(result, list): 
                    result = np.mean(result)
                line += f",{result}"
            
            line += "\n" 
            f.write(line)


        # Checkpoint loop
        if epoch % CKPT_EPOCH == 0:
            t = time.localtime()
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)
            save_checkpoint(
                CKPT_DIR,
                model,
                optimizer,
                epoch,
                date=current_time,
                model_name=None,
                name=f"model_{epoch}",
            )

        # Save checkpoints
        save_checkpoint(
            CKPT_DIR,
            model,
            optimizer,
            epoch,
            date=None,
            model_name=None,
            name="last_model",
        )


        # save best model and write logs to csv file
        
        if loss < best_loss:
            best_loss = loss
            
            save_checkpoint(
                CKPT_DIR,
                model,
                optimizer,
                epoch,
                date=None,
                model_name=None,
                name="best_model",
            )

            with open(os.path.join(OUT_DIR, "best_logs.csv"), "a") as f:

                f.write(line)



        # Validation loop
        with torch.no_grad():

            for i, data in enumerate(val_loader):

                print("\r", end=f'{progress_bar(i, length=75, train_set_len=len(val_loader)*VAL_BS, train_bs=VAL_BS)}')
                logging.info(f"Validation Epoch {epoch+1}, batch {i+1} / {len(val_loader)}")


                out = model(data)
                
                y_pred, confidences = out[:, :-N_TRAJ], out[:, -N_TRAJ:]
                y_pred = y_pred.view(-1, N_TRAJ, N_FUTURE, 2)

                gt = data[3]
                __results = eval_metrics(gt, y_pred, confidences, metrics)

                loss = loss_criteria(gt, y_pred, confidences)

                for name, result in __results.items(): 
                    if isinstance(result, list): 
                        result = np.mean(result)
                    writer.add_scalar(f'{name}/val', result, tb_it_v)

                writer.add_scalar('loss/val', loss, tb_it_v)
                tb_it_v += 1

if __name__=='__main__': 
    if EXPERIMENT_NAME=='Argo-1' : 
        trainset = Vectorset(TRAIN_DIR, normalize=False)
        valset = Vectorset(VAL_DIR, normalize=False)
    else :
        trainset = Vectorset(TRAIN_DIR, normalize=True)
        valset = Vectorset(VAL_DIR, normalize=True)
    
    model = VectorNet()
    trainloader = DataLoader(trainset, batch_size=TRAIN_BS, shuffle=True, collate_fn=custom_collate)
    valloader = DataLoader(valset, batch_size=VAL_BS, shuffle=True, collate_fn=custom_collate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = pytorch_neg_multi_log_likelihood_batch()

    # cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # tensorboard
    writer = SummaryWriter(log_dir=TB_DIR)

    # metrics
    metrics = {
        "nll": pytorch_neg_multi_log_likelihood_batch(),
        # "mse": torch.nn.MSELoss(),
        # "mae": torch.nn.L1Loss(),
        "mde": mean_displacement_error, 
        "fde": final_displacement_error, 
    }

    train_from_last_ckpt(model,
                            trainloader,
                            valloader,
                            optimizer,
                            loss_func,
                            scheduler,
                            CKPT_DIR,
                            writer,
                            metrics)
