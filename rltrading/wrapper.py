from typing import Tuple
from typing import Tuple
import gym
import numpy as np


class TradingWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env:gym.Env, seed:int=0):
    # Call the parent constructor, so we can access self.env later
    super().__init__(env)
    self.rng = np.random.default_rng(seed)
    self.mode = "train"
  
  def reset(self) -> np.ndarray:
    """
    Reset the environment 
    """
    if self.mode == "eval":
        self.env.current_step = self.env.history_length
    elif self.mode == "train":
        # Set the current step to a random point within the data frame
        self.env.current_step = self.rng.integers(low=self.env.history_length, 
                                                  high=len(self.env.df) - 1)
    else:
        raise ValueError("Mode must be either 'train' or 'eval'")

    obs = self.env.reset()
    return obs

  def step(self, action:int) -> Tuple[np.ndarray, float, bool, dict]:
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info
