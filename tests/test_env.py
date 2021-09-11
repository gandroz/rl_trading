import pandas as pd
from rltrading.env import StockTradingEnv


def test_new_observation():
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    env = StockTradingEnv(df)
    obs = env._next_observation()
    assert True