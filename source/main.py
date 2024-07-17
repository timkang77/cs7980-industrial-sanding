from dqn import *
from run_checkpoint import *
import numpy as np
from ray.tune.registry import register_env
from env import GridWorldEnv

np.random.seed(42)
test_grid = np.random.uniform(0, 4, (50, 50))

def env_creator(env_config):
        return GridWorldEnv(test_grid, render_mode = None)

def main():
    run_dqn(env_creator)
    env = env_creator({"render_mode":"human"})
   
if __name__ == "__main__":
    main()
    register_env("GridWorldEnv", env_creator)
    # run_checkpoint(env= env_creator({"render_mode":"human"})
    #               ,result_path=r"C:\Users\Jixian\Documents\cs7980-industrial-sanding\DQN_2024-07-10_11-27-14\DQN_GridWorldEnv_07f95_00009_9_2024-07-10_11-27-14\checkpoint_000000")