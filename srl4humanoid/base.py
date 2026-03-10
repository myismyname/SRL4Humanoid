"""
State Representation Learning (SRL) Base Module

This module defines the base interface for state representation learning algorithms,
providing a unified abstract base class for specific SRL algorithm implementations.
All SRL algorithms (e.g., PVP, SimSiam, SPR, VAE) should inherit from this base class.
"""

import torch.nn as nn


class SRL(nn.Module):
    """
    State Representation Learning Base Class
    
    This class defines the basic interface for all SRL algorithms, including the encoder,
    action dimension, and configuration parameters. Subclasses need to implement
    compute_loss and update_misc methods to define specific loss computation and
    parameter update logic.
    
    Attributes:
        encoder_online (nn.Module): Online encoder network for encoding observed states
                                   into feature representations
        action_dim (int): Dimension of the action space
        srl_cfg (dict): Configuration dictionary for SRL algorithm containing various
                       hyperparameters
    """
    
    def __init__(
        self, 
        encoder_online,
        action_dim,
        srl_cfg
    ):
        """
        Initialize the SRL base class
        
        Args:
            encoder_online (nn.Module): Online encoder network
            action_dim (int): Dimension of the action space
            srl_cfg (dict): SRL configuration dictionary
        """
        super(SRL, self).__init__()
        self.encoder_online = encoder_online
        self.action_dim = action_dim
        self.srl_cfg = srl_cfg

    def compute_loss(self, *args, **kwargs):
        """
        Compute the SRL loss function
        
        This method must be implemented in subclasses to compute the loss for
        the specific SRL algorithm.
        
        Args:
            *args: Variable positional arguments, defined by subclass
            **kwargs: Variable keyword arguments, defined by subclass
            
        Returns:
            torch.Tensor: Loss value
        """
        pass

    def update_misc(self):
        """
        Update miscellaneous parameters or states
        
        This method is used to update states other than model parameters, such as
        target networks, learning rate decay, etc. Subclasses can override this
        method as needed.
        """
        pass