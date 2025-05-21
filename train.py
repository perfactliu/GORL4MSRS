import os.path
import numpy as np
from arguments import get_args
from rl_modules.SAC_agent import Agent
import torch
from environments.environment import SatelliteEnv
import random


def get_env_params(argus, env):
    params = {'obs': 3,  # observation dim
              'goal': 3,  # goal dim
              'action': 48*env.satellite_number+1,  # action dim, 1代表全局不动
              'done': 1,
              'sat_num': env.satellite_number,
              'max_timestep': argus.max_cycle_steps
              }
    return params


def launch(argus):
    # create the env
    env = SatelliteEnv(argus.seed, argus.change_goal_cycle)
    # set random seeds for reproduce
    if argus.seed is not None:
        random.seed(argus.seed)
        np.random.seed(argus.seed)
        torch.manual_seed(argus.seed)

    # get the environment parameters
    env_params = get_env_params(argus, env)
    # create the sac agent to interact with the environment
    sac_trainer = Agent(args, env, env_params)
    sac_trainer.train_cycle()
    os.makedirs("checkpoints", exist_ok=True)
    sac_trainer.saveCheckpoints(os.path.join("checkpoints", argus.model_name))


if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
