from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune
import numpy as np
import ray
from env import GridWorldEnv


# Define a sample grid for initialization
test_grid = np.array([[3.178, 2.347, 9.928, 3.101, 3.328],
                     [5.749, 2.433, 6.492, 8.693, 3.472],
                     [6.866, 2.371, 6.783, 2.230, 6.691],
                     [2.632, 3.352, 7.248, 4.535, 4.740],
                     [7.143, 6.891, 9.766, 7.620, 3.492]])

# Register the environment
def env_creator(env_config):
    return GridWorldEnv(test_grid, render_mode = None)

tune.register_env("GridWorldEnv", env_creator)
ray.shutdown()  
ray.init()

# Configure the DQN algorithm
config = DQNConfig()

replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

config = config.training(replay_buffer_config=replay_config)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
config = config.environment(env="GridWorldEnv")

from ray import tune, train
tuner = tune.Tuner(
    "DQN",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        stop={"training_iteration": 20000,
              "env_runners/episode_return_mean": 350},
        checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
    ),
)
results = tuner.fit()
best_result = results.get_best_result(
    metric="env_runners/episode_return_mean", mode="max"
)
best_checkpoint = best_result.checkpoint
from ray.rllib.algorithms.algorithm import Algorithm
best_dqn = Algorithm.from_checkpoint(best_checkpoint)


'''# Evaluate the final variance
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
'''