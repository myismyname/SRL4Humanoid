# PPO
cuda_idx=1
CUDA_VISIBLE_DEVICES=${cuda_idx} python scripts/rsl_rl/train.py --headless \
    --logger wandb \
    --log_project_name srl4humanoid_g1 \
    --task Unitree-G1-29dof-Velocity \
    --seed 42 \
    --srl_algo_name ppo \
    > logs_g1/misc/g1_seed_42.log 2>&1 &

# PPO+PvP
cuda_idx=1
CUDA_VISIBLE_DEVICES=${cuda_idx} python scripts/rsl_rl/train.py \
    --logger wandb \
    --log_project_name srl4humanoid_g1 \
    --srl_algo_name ppo_pvp \
    --task Unitree-G1-29dof-Velocity \
    --headless \
    --pvp_hidden_dim 512 \
    --pvp_loss_coef 0.5 \
    --note int1 \
    --seed 42 \
    --srl_data_prop 2.0 \
    --srl_time_prop 50000 \
    --srl_interval 1 \
    > logs_g1/misc/g1_pvp0.5_cuda_${cuda_idx}_s42.log 2>&1 &

# PPO+SimSiam
cuda_idx=3
CUDA_VISIBLE_DEVICES=${cuda_idx} python scripts/rsl_rl/train.py \
    --srl_algo_name ppo_simsiam \
    --logger wandb \
    --log_project_name srl4humanoid_g1 \
    --task Unitree-G1-29dof-Velocity \
    --headless \
    --simsiam_hidden_dim 512 \
    --simsiam_loss_coef 0.5 \
    --simsiam_q_aug_type mask \
    --simsiam_k_aug_type none \
    --note none \
    --seed 42 \
    --srl_data_prop 2.0 \
    --srl_time_prop 50000 \
    --srl_interval 1 \
    > logs_g1/misc/g1_simsiam_cuda_${cuda_idx}_s42.log 2>&1 &


# PPO+SPR
cuda_idx=4
CUDA_VISIBLE_DEVICES=${cuda_idx} python scripts/rsl_rl/train.py \
    --logger wandb \
    --log_project_name srl4humanoid_g1 \
    --srl_algo_name ppo_spr \
    --task Unitree-G1-29dof-Velocity \
    --headless \
    --spr_hidden_dim 512 \
    --spr_loss_coef 0.1 \
    --spr_k 5 \
    --spr_tau 0.0 \
    --spr_avg_loss \
    --spr_skip 1 \
    --spr_aug_type gaussian \
    --srl_data_prop 2.0 \
    --srl_time_prop 50000 \
    --srl_interval 1 \
    --seed 42 \
    --note none > logs_g1/misc/g1_spr_cuda_${cuda_idx}_s42.log 2>&1 &

# PPO+VAE
cuda_idx=5
CUDA_VISIBLE_DEVICES=${cuda_idx} python scripts/rsl_rl/train.py \
    --logger wandb \
    --log_project_name srl4humanoid_g1 \
    --srl_algo_name ppo_vae \
    --task Unitree-G1-29dof-Velocity \
    --headless \
    --vae_latent_dim 128 \
    --vae_hidden_dim 512 \
    --vae_loss_coef 0.1 \
    --vae_kld_weight 0.1 \
    --note none \
    --seed 42 \
    --srl_data_prop 2.0 \
    --srl_time_prop 50000 \
    --srl_interval 1 \
    > logs_g1/misc/g1_vae_cuda_${cuda_idx}_s42.log 2>&1 &


