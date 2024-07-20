from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune,train
import numpy as np
import ray
from env import GridWorldEnv
from ray.rllib.algorithms.algorithm import Algorithm

def run_dqn(env_creator):
    tune.register_env("GridWorldEnv", env_creator)
    ray.shutdown()  
    ray.init(ignore_reinit_error=True)

    # Configure the DQN algorithm
    config = DQNConfig()

    replay_config = {
            "type": "MultiAgentReplayBuffer",
            "capacity": 60000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

    config = config.training(replay_buffer_config=replay_config)
    config = config.training(train_batch_size=50,noisy = False)
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=1)
    config = config.environment(env="GridWorldEnv")
    config = config.rollouts(
        num_rollout_workers=10,  # Number of parallel rollout workers
        num_envs_per_worker=5,  # Number of environments per worker
        rollout_fragment_length=50,  # Number of steps per rollout fragment
    )

    tuner = tune.Tuner(
        "DQN",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=train.RunConfig(
            stop={"training_iteration": 20000,  
            },
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
            storage_path=r"C:\Users\Jixian\Documents\cs7980-industrial-sanding"
        ),
    )
    tuner.fit()
    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", mode="max"
    )

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    best_dqn = Algorithm.from_checkpoint(best_checkpoint)
    env = env_creator(None)

    # Reset the environment to get the initial observation
    obs, info = env.reset()

    # Run the policy in the environment
    done = False
    total_reward = 0

    print("")
    print("Initial variance is", info["variance"])
    print("")

    while not done:
        action = best_dqn.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        env.render()  
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        total_reward += reward
        print("total_reward",total_reward)
        print(info)
    env.close()
    ray.shutdown()  