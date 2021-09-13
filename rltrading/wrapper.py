import gym
import numpy as np


class TradingWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env, seed=0):
    # Call the parent constructor, so we can access self.env later
    super().__init__(env)
    self.rng = np.random.default_rng(seed)
    self.mode = "train"
  
  def reset(self):
    """
    Reset the environment 
    """
    if self.mode == "eval":
        self.env.current_step = 99
    elif self.mode == "train":
        # Set the current step to a random point within the data frame
        self.env.current_step = self.rng.integers(low=99, 
                                                  high=len(self.env.df) - 1)
    else:
        raise ValueError("Mode must be either 'train' or 'eval'")

    obs = self.env.reset()
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info
