"""
Simple Siamese (SimSiam) State Representation Learning Module

This module implements the SimSiam algorithm for self-supervised learning without
negative pairs. SimSiam learns representations by maximizing similarity between
different augmented views of the same input using a stop-gradient operation.

Reference: Chen & He, "Exploring Simple Siamese Representation Learning", CVPR 2021
"""

from torch import nn
from torch.nn import functional as F

from .base import SRL
from .data_augs import (random_masking, 
                        random_amplitude_scaling, 
                        gaussian_noise)


class SimSiam(SRL):
    """
    Simple Siamese Representation Learning (SimSiam)
    
    SimSiam learns invariant representations by maximizing agreement between
    two augmented views of the same input. It uses a predictor network and
    stop-gradient to prevent collapse without requiring negative pairs,
    momentum encoders, or large batch sizes.
    
    Attributes:
        loss_coef (float): Coefficient to scale the SimSiam loss
        q_aug_type (str): Augmentation type for query view ('gaussian', 'mask', 'ras', 'none')
        k_aug_type (str): Augmentation type for key view ('gaussian', 'mask', 'ras', 'none')
        hidden_dim (int): Hidden dimension for the predictor network
        predictor (nn.Sequential): Two-layer MLP predictor network
        simsiam_step (int): Training step counter
        aug_q (callable): Augmentation function for query view
        aug_k (callable): Augmentation function for key view
        params (list): List of trainable parameters
    """
    
    def __init__(self, 
                 encoder_online,
                 action_dim,
                 srl_cfg
                 ):
        """
        Initialize the SimSiam module
        
        Args:
            encoder_online (nn.Module): Online encoder network
            action_dim (int): Dimension of the action space
            srl_cfg (dict): Configuration dictionary containing:
                - simsiam_loss_coef: Loss coefficient
                - simsiam_q_aug_type: Augmentation type for query view
                - simsiam_k_aug_type: Augmentation type for key view
                - simsiam_hidden_dim: Hidden dimension for predictor
        """
        super().__init__(encoder_online, action_dim, srl_cfg)
        self.loss_coef = srl_cfg['simsiam_loss_coef']
        self.q_aug_type = srl_cfg['simsiam_q_aug_type']
        self.k_aug_type = srl_cfg['simsiam_k_aug_type']
        self.hidden_dim = srl_cfg['simsiam_hidden_dim']

        # Build a 2-layer predictor network
        # The predictor is key to SimSiam's success in preventing collapse
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder_online[-1].out_features, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(self.hidden_dim, self.encoder_online[-1].out_features)  # Output layer
        )

        self.simsiam_step = 0
        
        # Setup augmentation function for query view
        if self.q_aug_type == 'gaussian':
            self.aug_q = gaussian_noise
        elif self.q_aug_type == 'mask':
            self.aug_q = random_masking
        elif self.q_aug_type == 'ras':
            self.aug_q = random_amplitude_scaling
        elif self.q_aug_type == 'none':
            self.aug_q = lambda x: x
        else:
            raise NotImplementedError(f'Unknown q augmentation type for SimSiam: {self.q_aug_type}')
        
        # Setup augmentation function for key view
        if self.k_aug_type == 'gaussian':
            self.aug_k = gaussian_noise
        elif self.k_aug_type == 'mask':
            self.aug_k = random_masking
        elif self.k_aug_type == 'ras':
            self.aug_k = random_amplitude_scaling
        elif self.k_aug_type == 'none':
            self.aug_k = lambda x: x
        else:
            raise NotImplementedError(f'Unknown k augmentation type for SimSiam: {self.k_aug_type}')

        # Collect trainable parameters for the predictor
        self.params = list(self.predictor.parameters())

    def compute_loss(self, states):
        """
        Compute the SimSiam contrastive loss
        
        This method implements the core SimSiam algorithm:
        1. Generate two augmented views of the input
        2. Encode both views through the shared encoder
        3. Apply predictor to both encoded views
        4. Compute symmetric negative cosine similarity with stop-gradient
        
        Args:
            states (torch.Tensor): Input states [batch_size, state_dim]
        
        Returns:
            torch.Tensor: Scaled SimSiam loss value
        """
        # Apply augmentations to create two different views
        states_q = self.aug_q(states)
        states_k = self.aug_k(states)

        # Encode both views through the online encoder
        z1 = self.encoder_online(states_q)  # [batch_size, encoder_dim]
        z2 = self.encoder_online(states_k)  # [batch_size, encoder_dim]

        # Apply predictor to both encoded features
        p1 = self.predictor(z1)  # [batch_size, encoder_dim]
        p2 = self.predictor(z2)  # [batch_size, encoder_dim]

        # Normalize all features to unit sphere
        # This makes the loss equivalent to negative cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # Compute symmetric negative cosine similarity loss
        # The detach() operation (stop-gradient) is crucial for preventing collapse
        # It ensures gradients only flow through the predictor path
        loss = -(F.cosine_similarity(p1, z2.detach()).mean() + 
                 F.cosine_similarity(p2, z1.detach()).mean()) * 0.5

        return loss * self.loss_coef
    
    def update_misc(self):
        """
        Update miscellaneous parameters
        
        Currently no additional updates are needed for SimSiam.
        """
        pass