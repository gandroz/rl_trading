import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from env import StockTradingEnv
from wrapper import TradingWrapper
from plot import plot_summary
import pandas as pd

df = pd.read_csv('./data/AAPL.csv')

# The algorithms require a vectorized environment to run
env = TradingWrapper(StockTradingEnv(df, gamma=0.95))
env = DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
model.save("./data/model")

history = []
done = False
env.mode = "eval"
obs = env.reset()
res = env.render()
while not done:
    action, _states = model.predict(obs)
    res.update({"action": StockTradingEnv.actions_str[action[0]]})
    history.append(res)
    obs, rewards, done, info = env.step(action)
    res = env.render()

history = pd.DataFrame(history)
history.to_csv("./data/res_APPL.csv")

print(f'Balance: {history.iloc[-1].balance}')
print(f'Shares held: {history.iloc[-1].shares_held}')
print(f'Net worth: {history.iloc[-1].net_worth}')
print(f'Profit: {history.iloc[-1].profit}')

plot_summary(df, history)
