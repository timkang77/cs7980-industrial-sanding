from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune,train
import numpy as np
import ray
from env import GridWorldEnv

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
    config = config.training(train_batch_size=400,noisy = False)
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=1)
    config = config.environment(env="GridWorldEnv")

    tuner = tune.Tuner(
        "DQN",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=train.RunConfig(
            stop={"training_iteration": 200,  
                "env_runners/episode_return_mean": 350},
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
            storage_path=r"C:\Users\Jixian\Documents\cs7980-industrial-sanding"
        ),
    )
    tuner.fit()
    ray.shutdown()  