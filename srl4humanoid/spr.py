"""
Self-Predictive Representations (SPR) State Representation Learning Module

This module implements the SPR algorithm, which learns representations by predicting
future latent states given current observations and actions. SPR combines temporal
prediction with contrastive learning using momentum encoders.

Reference: Schwarzer et al., "Data-Efficient Reinforcement Learning with
Self-Predictive Representations", ICLR 2021
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .base import SRL
from .data_augs import (random_masking, 
                        random_amplitude_scaling, 
                        gaussian_noise)


class SPR(SRL):
    """
    Self-Predictive Representations (SPR)
    
    SPR learns representations by predicting future encoded states in latent space
    given the current state and action sequence. It uses:
    1. A transition model to predict future latent states
    2. A momentum encoder (target) to encode actual future states
    3. A projector and predictor for contrastive learning
    
    Attributes:
        init_K (int): Initial prediction horizon (number of steps to predict)
        K (int): Current prediction horizon
        loss_coef (float): Coefficient to scale the SPR loss
        tau (float): Momentum coefficient for target network updates (EMA)
        avg_loss (bool): Whether to average loss over prediction steps
        loss_decay (bool): Whether to decay the loss coefficient over time
        skip_step (int): Number of steps to skip between predictions
        hidden_dim (int): Hidden dimension for networks
        encoder_target (nn.Module): Target encoder (momentum encoder)
        encoder_output_dim (int): Output dimension of the encoder
        proj_online (nn.Linear): Online projection head
        proj_target (nn.Linear): Target projection head (momentum)
        predictor (nn.Sequential): Predictor network
        transition (nn.Sequential): Transition model for predicting future states
        spr_step (int): Training step counter
        aug (callable): Data augmentation function
        params (list): List of trainable parameters
    """
    
    def __init__(self, 
                 encoder_online,
                 action_dim,
                 srl_cfg
                 ):
        """
        Initialize the SPR module
        
        Args:
            encoder_online (nn.Module): Online encoder network
            action_dim (int): Dimension of the action space
            srl_cfg (dict): Configuration dictionary containing:
                - spr_k: Prediction horizon (number of future steps)
                - spr_loss_coef: Loss coefficient
                - spr_tau: Momentum coefficient for target updates
                - spr_avg_loss: Whether to average loss over steps
                - spr_loss_decay: Whether to decay loss coefficient
                - spr_skip: Number of steps to skip between predictions
                - spr_hidden_dim: Hidden dimension
                - spr_aug_type: Augmentation type ('mask', 'random', 'gaussian', 'none')
        """
        super().__init__(encoder_online, action_dim, srl_cfg)
        self.init_K = srl_cfg['spr_k']
        self.K = srl_cfg['spr_k']
        self.loss_coef = srl_cfg['spr_loss_coef']
        self.tau = srl_cfg['spr_tau']
        self.avg_loss = srl_cfg['spr_avg_loss']
        self.loss_decay = srl_cfg['spr_loss_decay']
        self.skip_step = srl_cfg['spr_skip']

        self.hidden_dim = srl_cfg['spr_hidden_dim']

        # Get augmentation type from config
        spr_aug_type = srl_cfg['spr_aug_type']

        # Create target encoder as a copy of online encoder (momentum encoder)
        self.encoder_target = deepcopy(self.encoder_online)
        try:
            self.encoder_output_dim = self.encoder_online[-1].out_features
        except:
            self.encoder_output_dim = 32
        # Freeze target encoder parameters (will be updated via EMA)
        for param in self.encoder_target.parameters():
            param.requires_grad = False

        # Projection heads for contrastive learning
        self.proj_online = nn.Linear(self.encoder_output_dim, self.hidden_dim//2)
        self.proj_target = deepcopy(self.proj_online)
        # Freeze target projector parameters (will be updated via EMA)
        for param in self.proj_target.parameters():
            param.requires_grad = False

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim//2, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//2)
        )

        # Transition model: predicts next latent state given current latent and action
        self.transition = nn.Sequential(
            nn.Linear(self.encoder_output_dim + action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.encoder_output_dim)
        )

        # Collect trainable parameters
        self.params = list(self.proj_online.parameters()) + \
                      list(self.predictor.parameters()) + \
                      list(self.transition.parameters())

        self.spr_step = 0
        
        # Setup data augmentation function
        if spr_aug_type == 'mask':
            self.aug = random_masking
        elif spr_aug_type == 'random':
            self.aug = random_amplitude_scaling
        elif spr_aug_type == 'gaussian':
            self.aug = gaussian_noise
        elif spr_aug_type == 'none':
            self.aug = lambda x: x
        else:
            raise NotImplementedError(f"Unknown SPR augmentation type: {spr_aug_type}")

    def compute_loss(self, states, actions):
        """
        Compute the SPR loss by predicting future latent states
        
        This method implements the core SPR algorithm:
        1. Encode the initial state using the online encoder
        2. Use the transition model to predict future latent states given actions
        3. Encode actual future states using the target encoder
        4. Compute contrastive loss between predicted and actual future states
        
        Args:
            states (torch.Tensor): State sequence [batch_size, K+1, state_dim]
                                  where K is the prediction horizon
            actions (torch.Tensor): Action sequence [batch_size, K, action_dim]
        
        Returns:
            torch.Tensor: Scaled SPR loss value
        """
        # Apply data augmentation to states
        states = self.aug(states)
        
        # Encode the initial state
        z0 = self.encoder_online(states[:, 0])

        # Multi-step prediction: use transition model to predict future latent states
        preds = []
        z_prev = z0
        for k in range(0, self.K, self.skip_step):
            action = actions[:, k]  # [batch_size, action_dim]
            combined = torch.cat([z_prev, action], dim=1)  # [batch_size, encoder_dim+action_dim]
            z_pred = self.transition(combined)
            preds.append(z_pred)
            z_prev = z_pred
        
        # Get target representations: encode actual future states
        targets = []
        for k in range(1, self.K + 1, self.skip_step):
            with torch.no_grad():
                zt = self.encoder_target(states[:, k])  # [batch_size, encoder_dim]
                targets.append(zt)

        spr_loss = 0.
        
        # Compute SPR loss for each prediction step
        for z_pred, zt in zip(preds, targets):
            # Project and predict online features
            y_pred = self.predictor(self.proj_online(z_pred))
            y_pred = F.normalize(y_pred, dim=1)

            # Project target features (no gradient)
            with torch.no_grad():
                yt = self.proj_target(zt)
                yt = F.normalize(yt, dim=1)

            # Compute MSE loss between normalized representations
            # This is equivalent to negative cosine similarity
            spr_loss += F.mse_loss(yt, y_pred, reduction="none").sum(-1).mean(0)

        # Optionally average loss over prediction horizon
        if self.avg_loss:
            return spr_loss / self.K * self.loss_coef
        else:
            return spr_loss * self.loss_coef
    
    def update_target_network(self):
        """
        Update target networks using exponential moving average (EMA)
        
        This method updates the target encoder and projector using momentum updates:
        θ_target = τ * θ_target + (1 - τ) * θ_online
        
        This helps stabilize training by providing slowly changing targets.
        """
        with torch.no_grad():
            # Update target encoder parameters
            for param_o, param_t in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
                param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data

            # Update target projector parameters
            for param_o, param_t in zip(self.proj_online.parameters(), self.proj_target.parameters()):
                param_t.data = self.tau * param_t.data + (1 - self.tau) * param_o.data
        
        self.spr_step += 1
    
    def update_misc(self):
        """
        Update miscellaneous parameters
        
        This method updates the target networks and optionally decays the loss coefficient.
        """
        self.update_target_network()
        if self.loss_decay:
            # Gradually decay the loss coefficient
            self.loss_coef = self.loss_coef * 0.999