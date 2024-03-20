import numpy as np

import torch
from torch import nn

class pytorch_log_mean_displacement_error(nn.Module):
    """
        Compute the mean displacement error between the ground truth and the prediction.
    """ 
    def __init__(self):
        super(pytorch_log_mean_displacement_error, self).__init__()
    

    def forward(self, y, y_pred): 
        """
        Args:
            y (Tensor): array of shape (bs)x(time)x(2D coords)
            y_pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
            confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
            avails (Tensor): array of shape (bs)x(time) with the availability for each y timestep
        Returns:
            Tensor: negative log-likelihood for this example, a single float number
        """

        # convert to (batch_size, num_modes, future_len, num_coords)
        y = torch.unsqueeze(y, 1)  # add modes

        # error (batch_size, num_modes, future_len)
        error = torch.sum(
            ((y - y_pred)) ** 2, dim=-1
        )  # reduce coords and use availability


        # error (batch_size, num_modes)
        error = -torch.logsumexp(error, dim=-1, keepdim=True)

        return torch.mean(error)


class pytorch_neg_multi_log_likelihood_batch(nn.Module):
    
    def __init__(self):
        super(pytorch_neg_multi_log_likelihood_batch, self).__init__()
    
    def forward(self, y, y_pred, confidences): 
            """
            Compute a negative log-likelihood for the multi-modal scenario.
            Args:
                y (Tensor): array of shape (bs)x(time)x(2D coords)
                y_pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
                confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
                avails (Tensor): array of shape (bs)x(time) with the availability for each y timestep
            Returns:
                Tensor: negative log-likelihood for this example, a single float number
            """

            # convert to (batch_size bs, num_modes k, future_len T, num_coords 2)
            y = torch.unsqueeze(y, 1)  # add modes
            

            # error (batch_size, num_modes, future_len)
            error = torch.sum(
                (y - y_pred) ** 2, dim=-1
            )  # reduce coords and use availability
            
            with np.errstate(
                divide="ignore"
            ):  # when confidence is 0 log goes to -inf, but we're fine with it
                # error (batch_size, num_modes)
                error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
                    error, dim=-1
                )  # reduce time

            # error (batch_size, num_modes)
            error = -torch.logsumexp(error, dim=-1, keepdim=True)

            return torch.mean(error)

# Attention Penality
def attention_penalty(attention_matrix, target_distribution='uniform'):
    """
    Calculates the attention penalty based on the deviation from the target distribution.
    """
    batch_size, num_heads, seq_length, _ = attention_matrix.size()
    if target_distribution == 'uniform':
        # Uniform distribution: each element should have 1/seq_length attention
        target = torch.full_like(attention_matrix, fill_value=1/seq_length)
    else:
        raise ValueError("Unsupported target distribution")

    # Calculate the penalty as the mean squared error between attention and target
    penalty = torch.mean((attention_matrix - target) ** 2)
    return penalty



# Evaluation metrics
def mean_displacement_error(y, y_pred, conf):
    """
        Compute the final displacement error between the ground truth and the prediction.
    """
    error_ls = []
    for i in range(y_pred.shape[1]):
        error = ((y - y_pred[:, i])) ** 2
        error = torch.sum(torch.Tensor(error), dim=-1)
        error = torch.sqrt(error)
        error = error.view(-1)
        error_ls.append(np.mean(error.to('cpu').detach().numpy()))

    return error_ls


def final_displacement_error(y, y_pred, conf):
    """
        Compute the final displacement error between the ground truth and the prediction.
    """
    error_ls = []
    for i in range(y_pred.shape[1]):
        error = ((y[:, -1] - y_pred[:, i, -1])) ** 2
        error = torch.sum(torch.Tensor(error), dim=-1)  # reduce coords and use availability
        error = torch.sqrt(error)
        error_ls.append(torch.mean(error).item())
    return error_ls


def missrate(y_true, y_pred, heading, T, comb='avg') -> int:
    """
    Compute the miss rate between the ground truth and the prediction.

                        λlat    λlon
        T=3 seconds     1       2
        T=5 seconds     1.8     3.6
        T=8 seconds     3       6
    
    """
    R : np.ndarray = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    if comb == 'avg':
      MR = np.zeros((len(y_pred), T))
    elif comb=='min': 
      MR = np.ones((len(y_pred), T))
    for i in range(y_pred.shape[1]):
        err = y_true - y_pred[:, i]

        err = np.matmul(err, R) # shape -> bs * T * 2
        lat = [1, 1.8, 3]
        lon = [2, 3.6, 6]
        samples = [slice(30), slice(30, 50), slice(50, T)]

        for j in range(len(lat)):
            if comb=='min': 
              MR[:, samples[j]] = np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]) & (MR[:, samples[j]]==1), 1, 0)
            elif comb=='avg': 
              MR[:, samples[j]] += np.where((np.abs(err[:, samples[j], 0]) > lat[j]) | (np.abs(err[:, samples[j], 1]) > lon[j]), 1, 0)
        
        if comb=='avg': 
          MR = MR / y_pred.shape[1]
    return MR
