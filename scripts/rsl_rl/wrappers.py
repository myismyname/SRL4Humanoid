from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import gymnasium as gym
import torch

class UnitreeEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions):
        super().__init__(env, clip_actions=clip_actions)
        self.pvp_part_size = 15

    def get_observations(self):
        obs, extras = super().get_observations()
        extras["observations"]["pvp_part"] = extras["observations"]['critic'][:, :15]
        return obs, extras
    
    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        extras["observations"]["pvp_part"] = extras["observations"]['critic'][:, :15]
        return obs, rewards, dones, extras
    
    def reset(self):
        obs, extras = super().reset()
        extras["observations"]["pvp_part"] = extras["observations"]['critic'][:, :15]
        return obs, extras



class UnitreeEnvWrapperPadding(UnitreeEnvWrapper):
    def __init__(self, env, clip_actions):
        super().__init__(env, clip_actions=clip_actions)
        self.padding_size = self.pvp_part_size
        self.zeros = None

    def get_observations(self):
        obs, extras = super().get_observations()
        if self.zeros is None:
            self.zeros = torch.zeros(obs.shape[0], self.padding_size, device=obs.device, dtype=obs.dtype)
        obs = torch.cat([obs, self.zeros], dim=1)
        return obs, extras
    
    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        obs = torch.cat([obs, self.zeros], dim=1)
        return obs, rewards, dones, extras
    
    def reset(self):
        obs, extras = super().reset()
        obs = torch.cat([obs, self.zeros], dim=1)
        return obs, extras
