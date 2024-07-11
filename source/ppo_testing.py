
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env

import numpy as np

test_grid = np.array([[0.178, 2.347, 9.928, 0.101, 0.328],
                     [5.749, 2.433, 6.492, 8.693, 3.472],
                     [6.866, 2.371, 6.783, 2.230, 6.691],
                     [2.632, 3.352, 7.248, 4.535, 4.740],
                     [7.143, 6.891, 9.766, 7.620, 3.992]])

from env import GridWorldEnv


# Define the custom environment
def env_creator(env_config):
    return GridWorldEnv(test_grid, render_mode = env_config["render_mode"])

# Register the custom environment
register_env("GridWorldEnv", env_creator)


from ray.rllib.algorithms.algorithm import Algorithm

# 2000 iterations to reduce 1/3 of the initial variance
#result_path = "/Users/tanjun/ray_results/PPO_2024-06-24_23-21-25/PPO_GridWorldEnv_26967_00000_0_2024-06-24_23-21-25/checkpoint_000000"

# 20000 iterations to reduce 1/2 of the initial variance
result_path = "/Users/tanjun/ray_results/PPO_2024-06-25_11-04-56/PPO_GridWorldEnv_6e304_00000_0_2024-06-25_11-04-56/checkpoint_000000"
best_ppo = Algorithm.from_checkpoint(result_path)


# Initialize the environment
env = env_creator({"render_mode":"human", "size":5})

# Reset the environment to get the initial observation
obs, info = env.reset()

# Run the policy in the environment
done = False
total_reward = 0

print("")
print("Initial variance is", info["variance"])
print("")

while not done:
    action = best_ppo.compute_single_action(obs, explore=False)
    obs, reward, done, truncated, info = env.step(action)
    env.render()  
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    total_reward += reward
    print("total_reward",total_reward)
    print(info)
env.close()

