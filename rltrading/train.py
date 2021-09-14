from typing import final
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from env import StockTradingEnv
from wrapper import TradingWrapper
from scheduler import exponential_schedule
import pandas as pd

df = pd.read_csv('./data/AAPL.csv')

# The algorithms require a vectorized environment to run
env = TradingWrapper(StockTradingEnv(df, gamma=0.95))
env = DummyVecEnv([lambda: env])
lr_schedule = exponential_schedule(1e-4, 1e-6)
model = PPO("MlpPolicy", env, learning_rate=lr_schedule, verbose=1)
try:
    model.learn(total_timesteps=1_000_000)
except KeyboardInterrupt:
    model.save("./data/model")
else:
    model.save("./data/model")
