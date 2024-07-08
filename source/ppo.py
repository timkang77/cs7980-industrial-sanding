
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

new_test_grid = np.array([[3.178, 2.347, 4.928, 1.101, 2.328],
                     [5.749, 2.433, 6.492, 8.693, 3.472],
                     [6.866, 5.371, 6.783, 2.230, 3.691],
                     [2.632, 3.352, 5.248, 4.535, 4.740],
                     [2.143, 6.891, 9.766, 7.620, 3.992]])



import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid, render_mode=None, size=5, sander_size = 1, sander_shape = "rectangular", range = 0, power = 0.5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.range = range
        self.power = power


        #self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "grid": spaces.Box(low=-100, high=10, shape=(size, size), dtype=float) 
            })
        self.grid = grid
        self.step_counter = 0
        self.direction = None
        self.initial_grid = grid # for the reset() function
        self.prev_var = np.var(grid)
        self.threshold = self.prev_var/2

        # We have 5 actions, corresponding to "right", "up", "left", "down", "hold"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        #self.reward = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        #self.render_mode = "human"

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "grid": self.grid}



    def _get_info(self):
        return {
            "variance": np.var(self.grid),
            "step": self.step_counter}


    def reset(self, seed=None, options=None):
        # No randomness needed

        # Choose the agent's location as (0,0)
        self._agent_location = np.array([0,4])
        self.grid = self.initial_grid.copy()
        self.step_counter = 0
        self.direction = None
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


# we compute reward based on:
# 1. change of directions : penalty = 5
# 2. change of variances of the grid
# 3. step taken
    def reward(self, direction_change,curr_var):
      coeff = 100
      if np.any(self.grid < 0):
        return -10000
      if direction_change:
        return -5 + coeff * (self.prev_var-curr_var) - 1
      else:
        return coeff * (self.prev_var-curr_var) - 1


# Step

    def update_grid(self):
        # for circle sander
        for i in range(-self.range, self.range+1):
          for j in range(-self.range, self.range+1):
            if abs(i) + abs(j) <= self.range and (self._agent_location[0] + j <= self.size -1)\
              and (self._agent_location[0] + j >= 0) and (self._agent_location[1] + i <= self.size -1)\
              and (self._agent_location[1] + i >= 0):
              self.grid[self._agent_location[1] + j, self._agent_location[0] + i] -= self.power
              #print(i, j)
              #print(self._agent_location[0] + j, self._agent_location[1] + i)
              #print(self.grid[self._agent_location[1] + j, self._agent_location[0] + i])

        return


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        direction_change = False
        if action!= self.direction and self.step_counter != 0 and (action != 4 and self.direction != 4):
          direction_change = True
        if action != 4:
          self.direction = action
        self.step_counter += 1

        curr_var = np.var(self.grid)
        #TBD
        self.update_grid()

        reward = self.reward(direction_change,curr_var)

        #self.reward += reward
        self.prev_var = curr_var

        # An episode is done iff the agent has reached the target
        terminated = (curr_var <= self.threshold)

        truncated = False

        if np.any(self.grid < 0):
            truncated = True
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

# %%
# Rendering
# ~~~~~~~~~

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  


        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # draw the grid values
        font = pygame.font.SysFont(None, 24)
        for i in range(self.size):
            for j in range(self.size):
                value = self.grid[i, j]
                text = font.render(f'{value:.2f}', True, (0, 0, 0))
                text_rect = text.get_rect(center=((j + 0.5) * pix_square_size, (i + 0.5) * pix_square_size))
                canvas.blit(text, text_rect)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

# %%
# Close
# ~~~~~

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env

# Define the custom environment
def env_creator(env_config):
    return GridWorldEnv(test_grid, render_mode = env_config["render_mode"])

# Register the custom environment
register_env("GridWorldEnv", env_creator)

'''
# Initialize Ray
ray.shutdown()  
ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="GridWorldEnv", env_config={"render_mode": None})
    .framework("torch")  
    .rollouts(num_rollout_workers=1, rollout_fragment_length=100)
    .training(gamma=0.99, lr=0.0003, num_sgd_iter=6, vf_loss_coeff=0.01, use_kl_loss=True, train_batch_size=400)
)



from ray import tune, train
# Set up the Tuner
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        stop={"training_iteration": 20000,
              "env_runners/episode_return_mean": 350},
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
from ray.rllib.algorithms.algorithm import Algorithm
best_ppo = Algorithm.from_checkpoint(best_checkpoint)

'''

from ray.rllib.algorithms.algorithm import Algorithm

# 2000 iterations to reduce 1/3 of the initial variance
#result_path = "/Users/tanjun/ray_results/PPO_2024-06-24_23-21-25/PPO_GridWorldEnv_26967_00000_0_2024-06-24_23-21-25/checkpoint_000000"

# 20000 iterations to reduce 1/2 of the initial variance
# result_path = "/Users/tanjun/ray_results/PPO_2024-06-25_11-04-56/PPO_GridWorldEnv_6e304_00000_0_2024-06-25_11-04-56/checkpoint_000000"
result_path = "/Users/tkang/Documents/Northeastern/CS7980/algo/cs7980-industrial-sanding/checkpoint_000000"
best_ppo = Algorithm.from_checkpoint(result_path)


# Initialize the environment
env = env_creator({"render_mode":"human"})

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

