from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune
import numpy as np
import ray

# Import your custom environment
from new_grid_env import GridWorldEnv

# Define a sample grid for initialization
new_grid = np.array([[3.178, 2.347, 9.928, 3.101, 3.328],
                     [5.749, 2.433, 6.492, 8.693, 3.472],
                     [6.866, 2.371, 6.783, 2.230, 6.691],
                     [2.632, 3.352, 7.248, 4.535, 4.740],
                     [7.143, 6.891, 9.766, 7.620, 3.492]])

# Register the environment
def env_creator(env_config):
    return GridWorldEnv(env_config)

tune.register_env("grid_world_env", env_creator)

ray.init()

# Configure the DQN algorithm
config = DQNConfig() \
    .environment(env="grid_world_env", env_config={"grid": new_grid.copy(), "size": 5, "range": 1, "power": 0.2}) \
    .rollouts(num_rollout_workers=2, create_env_on_local_worker=True)

# Initialize and train the DQN algorithm
trainer = config.build()

for i in range(10):
    result = trainer.train()
    print(f"Iteration: {i}, info: {result['info']}")

# Evaluate the final variance
env = GridWorldEnv({"grid": new_grid.copy(), "size": 5, "range": 1, "power": 0.5})
obs, info = env.reset()

done = False
while not done:
    action = trainer.compute_single_action(obs)
    print("action:  ",action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

final_variance = info["variance"]
num_step = info["step"]
print(f"Final variance: {final_variance}")
print(f"Number of Steps: {num_step}")

ray.shutdown()