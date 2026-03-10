"""
SRL Configuration Processing Module

This module provides utilities for processing command-line arguments and configurations
for State Representation Learning (SRL) algorithms integrated with PPO training.
It handles argument parsing and experiment name generation for various SRL methods.
"""

import argparse


def add_srl_args(parser: argparse.ArgumentParser):
    """
    Add State Representation Learning (SRL) arguments to the argument parser
    
    This function creates a dedicated argument group for SRL-related parameters,
    including general SRL settings and algorithm-specific hyperparameters for
    SPR, VAE, PvP, and SimSiam.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to add SRL arguments to
    """
    # Create a new argument group for SRL-related arguments
    arg_group = parser.add_argument_group("srl", description="Arguments for State Representation Learning.")
    
    # ========== General SRL Arguments ==========
    arg_group.add_argument(
        "--note",
        type=str,
        default=None,
        help="Additional note or description for the experiment"
    )
    arg_group.add_argument(
        "--srl_algo_name",
        type=str,
        default=None,
        help="Name of the SRL algorithm (e.g., 'ppo', 'ppo_spr', 'ppo_vae', 'ppo_pvp', 'ppo_simsiam')"
    )
    arg_group.add_argument(
        "--srl_time_prop",
        type=int,
        default=50000,
        help="Time proportion for using SRL loss during training"
    )
    arg_group.add_argument(
        "--srl_data_prop",
        type=float,
        default=2.0,
        help="Proportion of data to use for computing SRL loss"
    )
    arg_group.add_argument(
        "--srl_interval",
        type=int,
        default=1,
        help="Interval (in iterations) for computing SRL loss"
    )

    # ========== SPR (Self-Predictive Representations) Arguments ==========
    arg_group.add_argument(
        "--spr_hidden_dim",
        type=int,
        help="Hidden dimension for SPR networks (projector and transition model)"
    )
    arg_group.add_argument(
        "--spr_loss_coef",
        type=float,
        help="Loss coefficient to scale the SPR loss"
    )
    arg_group.add_argument(
        "--spr_k",
        type=int,
        help="Prediction horizon (number of future steps to predict) for SPR"
    )
    arg_group.add_argument(
        "--spr_tau",
        type=float,
        help="Momentum coefficient for target network updates in SPR (EMA parameter)"
    )
    arg_group.add_argument(
        "--spr_avg_loss",
        action="store_true",
        help="Whether to average SPR loss over the prediction horizon"
    )
    arg_group.add_argument(
        "--spr_loss_decay",
        action="store_true",
        help="Whether to apply exponential decay to SPR loss coefficient"
    )
    arg_group.add_argument(
        "--spr_aug_type",
        type=str,
        help="Data augmentation type for SPR ('mask', 'random', 'gaussian', 'none')"
    )
    arg_group.add_argument(
        "--spr_skip",
        type=int,
        default=1,
        help="Number of timesteps to skip between predictions in SPR"
    )

    # ========== VAE (Variational Autoencoder) Arguments ==========
    arg_group.add_argument(
        "--vae_latent_dim",
        type=int,
        help="Dimension of the latent space for VAE"
    )
    arg_group.add_argument(
        "--vae_hidden_dim",
        type=int,
        help="Hidden dimension for VAE decoder network"
    )
    arg_group.add_argument(
        "--vae_loss_coef",
        type=float,
        help="Loss coefficient to scale the total VAE loss"
    )
    arg_group.add_argument(
        "--vae_kld_weight",
        type=float,
        help="Weight for the KL divergence term in VAE loss"
    )

    # ========== PvP (Proprioceptive-Privileged Contrastive Learning) Arguments ==========
    arg_group.add_argument(
        "--pvp_hidden_dim",
        type=int,
        help="Hidden dimension for PvP predictor network"
    )
    arg_group.add_argument(
        "--pvp_loss_coef",
        type=float,
        help="Loss coefficient to scale the PvP loss"
    )

    # ========== SimSiam (Simple Siamese Networks) Arguments ==========
    arg_group.add_argument(
        "--simsiam_hidden_dim",
        type=int,
        help="Hidden dimension for SimSiam predictor network"
    )
    arg_group.add_argument(
        "--simsiam_q_aug_type",
        type=str,
        help="Augmentation type for query view in SimSiam ('gaussian', 'mask', 'ras', 'none')"
    )
    arg_group.add_argument(
        "--simsiam_k_aug_type",
        type=str,
        help="Augmentation type for key view in SimSiam ('gaussian', 'mask', 'ras', 'none')"
    )
    arg_group.add_argument(
        "--simsiam_loss_coef",
        type=float,
        help="Loss coefficient to scale the SimSiam loss"
    )


def get_exp_name_and_cfg(args_cli):
    """
    Generate experiment name and SRL configuration dictionary from CLI arguments
    
    This function processes command-line arguments to create a descriptive experiment
    name (for logging and checkpointing) and a configuration dictionary containing
    all relevant hyperparameters for the selected SRL algorithm.
    
    Args:
        args_cli (argparse.Namespace): Parsed command-line arguments containing
                                       algorithm selection and hyperparameters
    
    Returns:
        tuple: A tuple containing:
            - experiment_name (str): Formatted experiment name with key hyperparameters
            - srl_cfg (dict): Configuration dictionary with all SRL-related settings
    
    Raises:
        ValueError: If an unknown srl_algo_name is provided
    """
    # Extract the selected SRL algorithm name
    srl_algo_name = args_cli.srl_algo_name
    
    # ========== Generate Experiment Name ==========
    # Create a descriptive experiment name that includes key hyperparameters
    # This helps identify and organize experiments in the logging directory
    
    if srl_algo_name == 'ppo':
        # Vanilla PPO without SRL
        experiment_name = f'ppo_note_{args_cli.note}-s{args_cli.seed}'
        
    elif srl_algo_name == 'ppo_spr':
        # PPO with Self-Predictive Representations
        experiment_name = (
            f'spr_coef_{args_cli.spr_loss_coef}_'
            f'decay_{args_cli.spr_loss_decay}_'
            f'k_{args_cli.spr_k}_'
            f'tau_{args_cli.spr_tau}_'
            f'avgloss_{args_cli.spr_avg_loss}_'
            f'aug_{args_cli.spr_aug_type}_'
            f'note_{args_cli.note}-s{args_cli.seed}'
        )
        
    elif srl_algo_name == 'ppo_vae':
        # PPO with Variational Autoencoder
        experiment_name = (
            f'vae_coef_{args_cli.vae_loss_coef}_'
            f'klw_{args_cli.vae_kld_weight}_'
            f'ldim_{args_cli.vae_latent_dim}_'
            f'note_{args_cli.note}-s{args_cli.seed}'
        )
        
    elif srl_algo_name == 'ppo_pvp':
        # PPO with Proprioceptive-Privileged Contrastive Learning
        experiment_name = (
            f'pvp_coef_{args_cli.pvp_loss_coef}_'
            f'hd_{args_cli.pvp_hidden_dim}_'
            f'note_{args_cli.note}-s{args_cli.seed}'
        )
        
    elif srl_algo_name == 'ppo_simsiam':
        # PPO with Simple Siamese Networks
        experiment_name = (
            f'simsiam_coef_{args_cli.simsiam_loss_coef}_'
            f'hd_{args_cli.simsiam_hidden_dim}_'
            f'qaug_{args_cli.simsiam_q_aug_type}_'
            f'kaug_{args_cli.simsiam_k_aug_type}_'
            f'note_{args_cli.note}-s{args_cli.seed}'
        )
        
    else:
        raise ValueError(f"Unknown srl_algo_name: {srl_algo_name}")

    # ========== Build SRL Configuration Dictionary ==========
    # Create algorithm-specific configuration dictionary
    
    srl_cfg = {}
    
    if srl_algo_name == 'ppo':
        # No additional configuration needed for vanilla PPO
        pass
        
    elif srl_algo_name == 'ppo_spr':
        # SPR-specific hyperparameters
        srl_cfg = {
            'spr_hidden_dim': args_cli.spr_hidden_dim,
            'spr_loss_coef': args_cli.spr_loss_coef,
            'spr_k': args_cli.spr_k,
            'spr_tau': args_cli.spr_tau,
            'spr_avg_loss': args_cli.spr_avg_loss,
            'spr_loss_decay': args_cli.spr_loss_decay,
            'spr_aug_type': args_cli.spr_aug_type,
            'spr_skip': args_cli.spr_skip
        }
        
    elif srl_algo_name == 'ppo_vae':
        # VAE-specific hyperparameters
        srl_cfg = {
            'vae_latent_dim': args_cli.vae_latent_dim,
            'vae_hidden_dim': args_cli.vae_hidden_dim,
            'vae_kl_weight': args_cli.vae_kld_weight,
            'vae_loss_coef': args_cli.vae_loss_coef,
        }
        
    elif srl_algo_name == 'ppo_pvp':
        # PvP-specific hyperparameters
        srl_cfg = {
            'pvp_hidden_dim': args_cli.pvp_hidden_dim,
            'pvp_loss_coef': args_cli.pvp_loss_coef,
        }
        
    elif srl_algo_name == 'ppo_simsiam':
        # SimSiam-specific hyperparameters
        srl_cfg = {
            'simsiam_hidden_dim': args_cli.simsiam_hidden_dim,
            'simsiam_loss_coef': args_cli.simsiam_loss_coef,
            'simsiam_q_aug_type': args_cli.simsiam_q_aug_type,
            'simsiam_k_aug_type': args_cli.simsiam_k_aug_type,
        }
        
    else:
        raise ValueError(f"Unknown srl_algo_name: {srl_algo_name}")
    
    # Add common SRL settings to the configuration
    srl_cfg['srl_algo_name'] = srl_algo_name
    srl_cfg['srl_time_prop'] = args_cli.srl_time_prop
    srl_cfg['srl_data_prop'] = args_cli.srl_data_prop
    srl_cfg['srl_interval'] = args_cli.srl_interval

    return experiment_name, srl_cfg