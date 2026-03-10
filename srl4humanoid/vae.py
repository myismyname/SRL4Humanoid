"""
Variational Autoencoder (VAE) Module for Representation Learning

This module implements a VAE for state representation learning.
The VAE learns a probabilistic latent representation by encoding states into
a latent distribution and reconstructing them through a decoder.

Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
"""

import torch
from torch import nn
from torch.nn import functional as F

from .base import SRL


class VAE(SRL):
    """
    Variational Autoencoder (VAE)
    
    The VAE learns a compressed latent representation by encoding inputs into
    a probability distribution (mean and variance) and reconstructing the original
    encoder features from samples drawn from this distribution. The training
    objective balances reconstruction accuracy with a regularization term that
    encourages the latent distribution to be close to a standard normal distribution.
    
    Attributes:
        hidden_dim (int): Hidden dimension for the decoder network
        kld_weight (float): Weight for the KL divergence loss term
        loss_coef (float): Coefficient to scale the total VAE loss
        encoder_output_dim (int): Output dimension of the encoder
        fc_mu (nn.Linear): Linear layer to compute latent mean
        fc_var (nn.Linear): Linear layer to compute latent log variance
        decoder (nn.Sequential): Decoder network to reconstruct features
        params (list): List of trainable parameters
    """
    
    def __init__(self, 
                 encoder_online,
                 action_dim,
                 srl_cfg
                 ):
        """
        Initialize the VAE module
        
        Args:
            encoder_online (nn.Module): Online encoder network
            action_dim (int): Dimension of the action space
            srl_cfg (dict): Configuration dictionary containing:
                - vae_latent_dim: Dimension of the latent space
                - vae_hidden_dim: Hidden dimension for decoder
                - vae_kl_weight: Weight for KL divergence term
                - vae_loss_coef: Loss coefficient
        """
        super().__init__(encoder_online, action_dim, srl_cfg)
        latent_dim = srl_cfg['vae_latent_dim']
        self.hidden_dim = srl_cfg['vae_hidden_dim']
        self.kld_weight = srl_cfg['vae_kl_weight']
        self.loss_coef = srl_cfg['vae_loss_coef']

        # Get encoder output dimension
        try:
            self.encoder_output_dim = self.encoder_online[-1].out_features
        except:
            self.encoder_output_dim = 32
        
        # Linear layers for computing latent distribution parameters
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder network: reconstructs encoder features from latent code
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, self.encoder_output_dim)
        )

        # Collect trainable parameters
        self.params = list(self.decoder.parameters()) + \
                      list(self.fc_mu.parameters()) + \
                      list(self.fc_var.parameters())
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from the latent distribution
        
        This technique allows backpropagation through the stochastic sampling process
        by expressing the random sample as a deterministic function of the parameters
        and an independent random variable: z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian distribution [batch_size, latent_dim]
            logvar (torch.Tensor): Log variance of the latent Gaussian [batch_size, latent_dim]
        
        Returns:
            torch.Tensor: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_loss(self, states):
        """
        Compute the VAE loss (reconstruction + KL divergence)
        
        The VAE loss consists of two terms:
        1. Reconstruction loss: measures how well the decoder reconstructs the input
        2. KL divergence: regularizes the latent distribution to be close to N(0, 1)
        
        Args:
            states (torch.Tensor): Input states [batch_size, state_dim]
        
        Returns:
            torch.Tensor: Scaled VAE loss value
        """
        # Encode states to get features
        features = self.encoder_online(states)
        
        # Compute latent distribution parameters
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        
        # Sample from the latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Reconstruct features from latent code
        z_recon = self.decoder(z)

        # Reconstruction loss: MSE between original and reconstructed features
        recon_loss = F.mse_loss(z_recon, features, reduction='mean')
        
        # KL divergence loss: encourages latent distribution to be close to N(0, 1)
        # KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Combine both losses
        loss = recon_loss + self.kld_weight * kld_loss

        return loss * self.loss_coef