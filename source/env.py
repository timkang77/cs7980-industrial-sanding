import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

#test_grid = np.load('matrix_5x10.npy')


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid, render_mode=None, sander_shape = "rectangular", range = 0, power = 0.05):
        self.length, self.width = grid.shape
        
        self.window_size = 512

        self.range = range
        self.power = power

        self.observation_space = spaces.Dict({
            "agent": spaces.MultiBinary(self.length * self.width),
            "grid": spaces.Box(low = -1, high = 100, shape=(self.length*self.width, ), dtype=float),
            })
        #self.observation_space = spaces.Dict({
        #    "agent": spaces.Box(low = 0, high = np.array([self.width - 1, self.length - 1]), shape=(2,), dtype=int),
        #    "grid": spaces.Box(low = -1, high = 100, shape=(self.length*self.width, ), dtype=float),
        #    })
        
        self.grid = grid
        self.step_counter = 0
        self.direction = None
        self.initial_grid = grid # for the reset() function
        self.threshold_mean = 0.0505
        self.threshold_max = 0.11
        self.maxspot = np.max(self.grid)

        # We have 5 actions, corresponding to "right", "down", "left", "up", "hold"
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

        self.path = []

        self.window = None
        self.clock = None


    def _get_obs(self):
        #return {"agent": self._agent_location, "grid": self.grid.flatten()}
        agent_one_hot = np.zeros(self.length * self.width, dtype=int)
        agent_one_hot[self._agent_location[1] * self.width + self._agent_location[0]] = 1
        return {"agent": agent_one_hot, "grid": self.grid.flatten()}



    def _get_info(self):
        return {
            "mean": np.mean(self.grid),
            "max": np.max(self.grid),
            "step": self.step_counter}


    def reset(self, seed=None, options=None):

        # Choose the agent's location as (0,0)
        self._agent_location = np.array([0, 0])
        self.grid = self.initial_grid.copy()
        self.grid[0, 0] -= self.power
        self.step_counter = 0
        self.direction = None
        observation = self._get_obs()
        info = self._get_info()

        self.path = [self._agent_location.copy()]

        self.maxspot = np.max(self.grid)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

# Reward
# we compute reward based on:
# 1. change of directions : penalty = 5
# 2. change of variances of the grid
# 3. step taken
    def reward(self, direction_change, sanded):
      
      coeff = 10

      if np.any(self.grid < 0):
        return -5
      #if np.mean(self.grid) < self.threshold_mean:
      #  return 10000

      diff_mean = self.grid[self._agent_location[1], self._agent_location[0]] 

      step_reward = -0.02

      step_reward += diff_mean*coeff

      if direction_change:
        step_reward -= 0.03
      if np.max(self.grid) < self.threshold_max:
        step_reward += 5

      return step_reward


# Step
    def update_grid(self):
        # all squares within the range of the manhattan distance of the sander will be sanded
        for i in range(-self.range, self.range+1):
          for j in range(-self.range, self.range+1):
            if abs(i) + abs(j) <= self.range and (self._agent_location[0] + j <= self.width -1)\
              and (self._agent_location[0] + j >= 0) and (self._agent_location[1] + i <= self.length -1)\
              and (self._agent_location[1] + i >= 0):
              self.grid[self._agent_location[1] + j, self._agent_location[0] + i] -= self.power

        return


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = self._agent_location + direction
        self._agent_location[0] = np.clip(new_location[0], 0, self.width - 1)
        self._agent_location[1] = np.clip(new_location[1], 0, self.length - 1)
        direction_change = False
        if action!= self.direction and self.direction != None and (action != 4 and self.direction != 4):
          direction_change = True
        if action != 4:
          self.direction = action
        self.step_counter += 1

        sanded = False
        if self.grid[self._agent_location[1], self._agent_location[0]] > self.threshold_max:
           sanded = True
           
        self.update_grid()

        reward = self.reward(direction_change, sanded)

        curr_mean = np.mean(self.grid)
        curr_max = np.max(self.grid)
        
        # An episode is done iff the agent has reached the target
        terminated = (curr_max < self.threshold_max) 

        truncated = False
        if np.any(self.grid < 0):
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()

        self.path.append(self._agent_location.copy())

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
        
        pix_square_size = self.window_size / max(self.length, self.width)

        offset_x = (self.window_size - pix_square_size * self.width) / 2
        offset_y = (self.window_size - pix_square_size * self.length) / 2


 
        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (offset_x + (self._agent_location[0] + 0.5) * pix_square_size, offset_y + (self._agent_location[1] + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                start_pos = (offset_x + (self.path[i][0] + 0.5) * pix_square_size,
                             offset_y + (self.path[i][1] + 0.5) * pix_square_size)
                end_pos = (offset_x + (self.path[i + 1][0] + 0.5) * pix_square_size,
                           offset_y + (self.path[i + 1][1] + 0.5) * pix_square_size)
                pygame.draw.line(canvas, (255, 0, 0), start_pos, end_pos, width=2)

        # draw the grid values
        font = pygame.font.SysFont(None, 24)
        for i in range(self.length):
            for j in range(self.width):
                value = self.grid[i, j]
                text = font.render(f'{value:.2f}', True, (0, 0, 0))
                text_rect = text.get_rect(center=(offset_x + (j + 0.5) * pix_square_size, offset_y + (i + 0.5) * pix_square_size))
                canvas.blit(text, text_rect)

        # add some gridlines
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (offset_x + pix_square_size * x, offset_y),
                (offset_x + pix_square_size * x, offset_y + pix_square_size * self.length),
                width=3,
            )

        for y in range(self.length + 1):
            pygame.draw.line(
                canvas,
                0,
                (offset_x, offset_y + pix_square_size * y),
                (offset_x + pix_square_size * self.width, offset_y + pix_square_size * y),
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


'''
def generate_traversal_sequence(matrix_shape):
    rows, cols = matrix_shape
    actions = []
    current_direction = 0  # Start moving "right"
    for row in range(rows):
        if row % 2 == 0:  # If the row number is even, move "right"
            for col in range(cols - 1):
                actions.append(0)  # Move "right"
            if row != rows - 1:
                actions.append(1)  # Move "down" at the end of the row
        else:  # If the row number is odd, move "left"
            for col in range(cols - 1):
                actions.append(2)  # Move "left"
            if row != rows - 1:
                actions.append(1)  # Move "down" at the end of the row
    return actions

matrix_shape = (5, 10)
actions = generate_traversal_sequence(matrix_shape)

from gymnasium.envs.registration import register

register(
    id='GridWorldTest',
    entry_point='__main__:GridWorldEnv',
)

env = gym.make('GridWorldTest', render_mode = "human")

env.reset()

t_reward = 0

for action in actions:
    obs, reward, done, truncated, info  = env.step(action)
    env.render()
    print(info)
    t_reward += reward
print("Total reward:", t_reward)
env.close()
'''