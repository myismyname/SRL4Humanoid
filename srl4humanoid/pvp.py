"""
Proprioceptive-Privileged Contrastive Learning (PvP) Module

This module implements the PvP algorithm, which learns representations by maximizing
agreement between proprioceptive observations and privileged information.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .base import SRL


class PvP(SRL):
    """
    Proprioceptive-Privileged Contrastive Learning (PvP)
    
    PvP learns representations by contrasting proprioceptive observations with
    observations that include privileged information (e.g., terrain features,
    object states). This approach helps transfer knowledge from privileged
    information to proprioceptive encoders.
    
    Attributes:
        loss_coef (float): Coefficient to scale the PvP loss
        hidden_dim (int): Hidden dimension for the predictor network
        predictor (nn.Sequential): Two-layer MLP predictor network
        pvp_step (int): Training step counter
        params (list): List of trainable parameters
        padding_size (int): Size of padding for dimension matching (if needed)
    """
    
    def __init__(self, 
                 encoder_online,
                 action_dim,
                 srl_cfg
                 ):
        """
        Initialize the PvP module
        
        Args:
            encoder_online (nn.Module): Online encoder network
            action_dim (int): Dimension of the action space
            srl_cfg (dict): Configuration dictionary containing:
                - pvp_loss_coef: Loss coefficient
                - pvp_hidden_dim: Hidden dimension for predictor
        """
        super().__init__(encoder_online, action_dim, srl_cfg)
        self.loss_coef = srl_cfg['pvp_loss_coef']
        self.hidden_dim = srl_cfg['pvp_hidden_dim']

        # Build a 2-layer predictor network
        # This predictor maps encoded features to a prediction space
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder_online[-1].out_features, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(self.hidden_dim, self.encoder_online[-1].out_features)  # Output layer
        )

        self.pvp_step = 0
        
        # Collect trainable parameters for the predictor
        self.params = list(self.predictor.parameters())

        self.padding_size = None

    def compute_loss(self, proprioceptive_obs, privileged_obs):
        """
        Compute the PvP contrastive loss
        
        This method computes the loss by:
        1. Creating two views: one with proprioceptive obs, one with privileged info
        2. Encoding both views through the encoder
        3. Predicting one view from another using the predictor
        4. Computing negative cosine similarity between predictions and targets
        
        Args:
            proprioceptive_obs (torch.Tensor): Proprioceptive observations [batch_size, obs_dim]
            privileged_obs (torch.Tensor): Privileged observations [batch_size, priv_dim]
        
        Returns:
            torch.Tensor: Scaled PvP loss value
        """
        # Create query view from proprioceptive observations
        x_q = proprioceptive_obs
        pvp_part_dim = privileged_obs.shape[-1]
        
        # Create key view by concatenating proprioceptive (without last 20 dims) and privileged info
        # This creates an augmented view that includes privileged information
        x_k = torch.cat((torch.clone(proprioceptive_obs[:, :-pvp_part_dim]), privileged_obs), dim=-1)

        # Encode both views through the online encoder
        z1 = self.encoder_online(x_q)  # [batch_size, encoder_dim]
        z2 = self.encoder_online(x_k)  # [batch_size, encoder_dim]

        # Apply predictor to encoded features
        p1 = self.predictor(z1)  # [batch_size, encoder_dim]
        p2 = self.predictor(z2)  # [batch_size, encoder_dim]

        # Normalize all features to unit sphere
        # This makes the loss equivalent to negative cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # Compute symmetric negative cosine similarity loss
        # Detach targets to prevent collapse (stop-gradient)
        loss = -(F.cosine_similarity(p1, z2.detach()).mean() + 
                 F.cosine_similarity(p2, z1.detach()).mean()) * 0.5

        return loss * self.loss_coef
    
    def update_misc(self):
        """
        Update miscellaneous parameters
        
        Currently no additional updates are needed for PVP.
        """
        pass