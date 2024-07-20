from dqn import *
from run_checkpoint import *
import numpy as np
from ray.tune.registry import register_env
from env import GridWorldEnv
from process_ply import output_grid
from process_ply import global_variance

file_path = r"C:\Users\Jixian\Documents\cs7980-industrial-sanding\source\TableTop_half_randomized.ply"
num_cols = 10
num_rows = 5

test_grid = output_grid(file_path,num_cols,num_rows)
variance = global_variance(file_path)

def env_creator(env_config):
        print(test_grid)
        return GridWorldEnv(test_grid,variance,num_cols,num_rows,0.03)

def main():
    run_dqn(env_creator)
    env = env_creator(env_config=None)
   
if __name__ == "__main__":
    main()
        
    #register_env("GridWorldEnv", env_creator)
    #run_checkpoint(env= env_creator(None)
                   #,result_path=r"C:\Users\Jixian\Documents\cs7980-industrial-sanding\DQN_2024-07-20_00-43-24\DQN_GridWorldEnv_be8b0_00000_0_2024-07-20_00-43-24\checkpoint_000000")