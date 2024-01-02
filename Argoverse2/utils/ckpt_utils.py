import os
import torch
import logging


def save_checkpoint(checkpoint_dir, 
                    model, 
                    optimizer, 
                    scheduler,
                    end_epoch, 
                    date=None, model_name=None, name=None, save_scheduler=False):
    """Saves a checkpoint of a model and optimizer.

    Args:
        checkpoint_dir: The directory to save the checkpoint to.
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The scheduler to save.
        end_epoch: The epoch number of the checkpoint.
        date: The date of the checkpoint.
        model_name: The name of the model.

    Side effects:
        Creates a checkpoint file in the specified directory.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'end_epoch': end_epoch,
    }

    if save_scheduler:
        state['scheduler'] = scheduler.state_dict()
    
    if name is not None:
        checkpoint_path = os.path.join(checkpoint_dir, name)
        checkpoint_path = checkpoint_path + '.pth'
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.{date}.{model_name}.pth')

    torch.save(state, checkpoint_path)
    logging.info('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, save_scheduler=False):
    """Loads a checkpoint into a model and optimizer.

    Args:
        checkpoint_path: The path to the checkpoint file.
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.

    Returns:
        The epoch number of the checkpoint.
    """

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if save_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

    end_epoch = checkpoint['end_epoch']
    return end_epoch