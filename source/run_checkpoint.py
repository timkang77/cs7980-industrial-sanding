from ray import tune
from env import GridWorldEnv
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
from ray.tune.registry import register_env

def run_checkpoint(env, result_path):
    # Restore the best model
    best_dqn = Algorithm.from_checkpoint(result_path)

    # Reset the environment to get the initial observation
    obs, info = env.reset()

    # Open a log file to record actions
    with open('dqn_actions.log', 'w') as f:
        done = False
        total_reward = 0

        print("")
        print("Initial info is", info)
        print("")

        while not done:
            action = best_dqn.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            total_reward += reward
            print("total_reward", total_reward)
            print(info)

            # Log the action
            f.write(f"{action},")
        
        # Add a newline at the end of the log
        f.write("\n")

# Load the training matrix
test_grid = np.load('z_matrix.npy')
# Define the custom environment
def env_creator(env_config):
    return GridWorldEnv(test_grid, render_mode=env_config["render_mode"])

if __name__ == "__main__":
    result_path = r"C:\Users\Jixian\Documents\cs7980-industrial-sanding\DQN_2024-07-24_09-59-14\DQN_GridWorldEnv_0e887_00000_0_2024-07-24_09-59-14\checkpoint_000000"
    env = env_creator(env_config={"render_mode": None, "size": 50})
    register_env("GridWorldEnv", env_creator)
    run_checkpoint(env, result_path)