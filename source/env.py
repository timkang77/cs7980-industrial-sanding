import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid, render_mode=None, size=5, sander_shape = "rectangular", range = 0, power = 0.5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.range = range
        self.power = power

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "grid": spaces.Box(low=-100, high=100, shape=(size, size), dtype=float) 
            })
        self.grid = grid
        self.step_counter = 0
        self.direction = None
        self.initial_grid = grid # for the reset() function
        self.prev_var = np.var(grid)
        self.threshold = self.prev_var/2

        # We have 5 actions, corresponding to "right", "up", "left", "down", "hold"
        self.action_space = spaces.Discrete(5)

        # action space
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "grid": self.grid}



    def _get_info(self):
        return {
            "variance": np.var(self.grid),
            "step": self.step_counter}


    def reset(self, seed=None, options=None):

        # Choose the agent's location as (0,0)
        self._agent_location = np.array([0,self.size-1])
        self.grid = self.initial_grid.copy()
        self.step_counter = 0
        self.direction = None
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

# Reward
# we compute reward based on:
# 1. change of directions : penalty = 5
# 2. change of variances of the grid
# 3. step taken
    def reward(self, direction_change,curr_var):
      coeff = 100
      if np.any(self.grid < 0):
        return -10000
      if direction_change:
        return -50 + coeff * (self.prev_var-curr_var) - 10
      else:
        return coeff * (self.prev_var-curr_var) - 10


# Step
    def update_grid(self):
        # all squares within the range of the manhattan distance of the sander will be sanded
        for i in range(-self.range, self.range+1):
          for j in range(-self.range, self.range+1):
            if abs(i) + abs(j) <= self.range and (self._agent_location[0] + j <= self.size -1)\
              and (self._agent_location[0] + j >= 0) and (self._agent_location[1] + i <= self.size -1)\
              and (self._agent_location[1] + i >= 0):
              self.grid[self._agent_location[1] + j, self._agent_location[0] + i] -= self.power

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
        
        self.update_grid()

        reward = self.reward(direction_change,curr_var)

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

# Rendering
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

        # add some gridlines
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

# Close
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
