"""
Data Augmentation Module for State Representation Learning

This module provides various data augmentation functions commonly used in
state representation learning for humanoid robot control. These augmentation techniques
help improve the robustness and generalization of learned representations.
"""

import torch


def gaussian_noise(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Add Gaussian noise to input tensor for data augmentation
    
    This augmentation adds random Gaussian noise to the input, which can help
    the model learn more robust representations that are invariant to small
    perturbations in the input space.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, feature_dim]
        sigma (float, optional): Standard deviation of the Gaussian noise.
                                Defaults to 1.0
    
    Returns:
        torch.Tensor: Augmented tensor with the same shape as input
    """
    return x + torch.randn_like(x) * sigma


def random_amplitude_scaling(x: torch.Tensor, low: float = 0.6, high: float = 1.2) -> torch.Tensor:
    """
    Apply random amplitude scaling to input tensor
    
    This augmentation randomly scales the amplitude of each sample in the batch
    by a factor uniformly sampled from [low, high]. This helps the model learn
    representations that are invariant to amplitude variations.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, feature_dim]
        low (float, optional): Lower bound of the scaling factor. Defaults to 0.6
        high (float, optional): Upper bound of the scaling factor. Defaults to 1.2
    
    Returns:
        torch.Tensor: Scaled tensor with the same shape as input
    """
    scale = torch.empty(x.size(0), device=x.device).uniform_(low, high)
    return x * scale.unsqueeze(-1)  


def random_masking(x: torch.Tensor, mask_ratio: float = 0.1, mask_value: float = 0.0) -> torch.Tensor:
    """
    Apply random masking to input tensor
    
    This augmentation randomly masks out elements of the input tensor by setting
    them to a specified value. This encourages the model to learn representations
    that can handle partial observability.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, feature_dim]
        mask_ratio (float, optional): Probability of masking each element.
                                     Defaults to 0.1 (10% masking)
        mask_value (float, optional): Value to set for masked elements.
                                     Defaults to 0.0
    
    Returns:
        torch.Tensor: Masked tensor with the same shape as input
    """
    mask = (torch.rand_like(x) >= mask_ratio).float()
    return x * mask + (1 - mask) * mask_value