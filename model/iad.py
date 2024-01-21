import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from torch import Tensor


def iad_head(local_feature, global_feature):
    """
    Computes the Inconsistency Anomaly Detection (IAD) loss/score between the local and global features.
    
    Args:
    - local_feature: A tensor of local features.
    - global_feature: A tensor of global features.
    
    Returns:
    - A tensor representing the IAD loss/score.
    """
    # Ensure the local and global features have the same shape
    assert local_feature.shape == global_feature.shape, "Features must have the same shape"
    
    # Compute the squared L2 norm of the difference
    difference = local_feature - global_feature
    squared_l2_norm = torch.norm(difference, p=2, dim=1).pow(2)
    
    # Normalize by the number of features
    feature_dim = local_feature.size(1)
    l_iad = squared_l2_norm / feature_dim
    
    # Take the mean across the batch
    batch_mean_iad = l_iad.mean()
    
    return batch_mean_iad