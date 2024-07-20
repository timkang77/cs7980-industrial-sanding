import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

def calculate_diff_variance(location,power,prev_mean,curr_mean, mean_space,frequency_space):
   N = np.sum(frequency_space)
   return (-2)*frequency_space[location[0],location[1]]* power* (mean_space[location[0],location[1]] + power/2)/N + (prev_mean**2 - curr_mean**2)

def calculate_mean(mean_space,frequency_space):
    N = np.sum(frequency_space)
    return  np.sum(frequency_space * mean_space) / N

class GridWorldEnv(gym.Env):

    def __init__(self, grid,initial_var, lsize, wsize, power = 0.5):
        self.lsize = lsize
        self.wsize= wsize 
        self.initial_var = initial_var
        self.power = power
        self.initial_grid = grid

        #self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=np.array([0, 0]), high=np.array([lsize - 1, wsize - 1]), dtype=int),
            "mean_space": spaces.Box(low=-0.1, high=10, shape=(lsize, wsize), dtype=float),
            "frequency_space" : spaces.Box(low=0, high=1000, shape=(lsize, wsize), dtype=int),
            "min_value_space" : spaces.Box(low=-0.4, high=10, shape=(lsize, wsize), dtype=float)
        
            })
        
       

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

    def _get_obs(self):
        return {"agent": self._agent_location, "mean_space": self.mean_space, "frequency_space" :self.frequency_space,"min_value_space": self.min_value_space }



    def _get_info(self):
        return {
            "variance": self.curr_var,
            "step": self.step_counter}


    def reset(self, seed=None, options=None):

        # Choose the agent's location as (0,0)
        self._agent_location = np.array([0,0])
        self.grid = self.initial_grid.copy()

        # suppose each entry from the grid is of form (mean, frequency, min_value)
        self.mean_space = np.zeros((self.lsize, self.wsize), dtype=float)
        self.frequency_space = np.zeros((self.lsize, self.wsize), dtype=float)
        self.min_value_space = np.zeros((self.lsize, self.wsize), dtype=float)

        for i in range(self.lsize):
            for j in range(self.wsize):
                self.mean_space[i, j] = self.grid[i, j, 0]
                self.frequency_space[i, j] = self.grid[i, j, 1]
                self.min_value_space[i, j] = self.grid[i, j, 2]

    

        self.step_counter = 0
        self.direction = None

        self.prev_mean = calculate_mean(self.mean_space,self.frequency_space)
        self.curr_mean = self.prev_mean

        self.curr_var = self.initial_var
        self.threshold = self.initial_var/2

        observation = self._get_obs()
        info = self._get_info()


        return observation, info


# we compute reward based on:
# 1. change of directions : penalty = 5
# 2. change of variances of the grid
# 3. step taken
    def reward(self, direction_change,diff_var):
        coeff = 10000
        if np.any(self.min_value_space < 0):
            return -1000
        if direction_change:
            return -5 + coeff *(-diff_var) - 1
        else:
            return coeff *(-diff_var) - 1


# Step

    def update_grid(self):
  
        self.min_value_space[self._agent_location[0], self._agent_location[1]] -= self.power
        self.mean_space[self._agent_location[0], self._agent_location[1]] -= self.power

        return


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction,  [0, 0], [self.lsize - 1, self.wsize - 1])
        #print("movement:  ",direction)
        direction_change = False
        if action!= self.direction and self.step_counter != 0 and (action != 4 and self.direction != 4):
          direction_change = True
        if action != 4:
          self.direction = action
        self.step_counter += 1

        self.prev_mean = self.curr_mean
        self.update_grid()
        self.curr_mean = calculate_mean(self.mean_space,self.frequency_space)
        diff_var =  calculate_diff_variance(self._agent_location,self.power,self.prev_mean,self.curr_mean,self.mean_space,self.frequency_space)
        
        self.curr_var += diff_var
        reward = self.reward(direction_change,diff_var)

    

        # An episode is done iff the agent has reached the target
        truncated = False
        terminated = False
        print("diff_var=  ",diff_var )
        print("curr var= ", self.curr_var)
        print("threshold = ",self.threshold)
        terminated = (self.curr_var <= self.threshold) 
        print("ter???", terminated)
        if np.any(self.min_value_space < 0):
            truncated = True
        print("trunc??,", truncated)

    
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


    def render(self):
        return None

    

    def close(self):
        return None