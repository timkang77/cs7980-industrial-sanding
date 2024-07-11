
import ray
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env

from env import GridWorldEnv
import numpy as np

from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm


test_grid = random_matrix = np.load('training_matrix.npy')

# Define the custom environment
def env_creator(env_config):
    return GridWorldEnv(test_grid, render_mode = env_config["render_mode"], size = env_config["size"])

# Register the custom environment
register_env("GridWorldEnv", env_creator)

# Initialize Ray
ray.shutdown()  
ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="GridWorldEnv", env_config={"render_mode": None, "size": 50})
    .framework("torch")  
    .rollouts(num_rollout_workers=1, rollout_fragment_length=100)
    .training(gamma=0.99, lr=0.0003, num_sgd_iter=6, vf_loss_coeff=0.01, use_kl_loss=True, train_batch_size=400)
)

# Set up the Tuner
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        stop={"training_iteration": 50000},
        checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
    ),
)

# Fit the model
results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(
    metric="env_runners/episode_return_mean", mode="max"
)

# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint

# Restore the best model
best_ppo = Algorithm.from_checkpoint(best_checkpoint)

# Initialize the environment
env = env_creator({"render_mode":"human", "size":50})

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