import pandas as pd
from pytest import approx
from rltrading.env import StockTradingEnv, MAX_SHARE_PRICE, MAX_VOL_SHARES


def test_new_observation():
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    history_length = 10
    env = StockTradingEnv(df, history_length=history_length)
    env.current_step = 100
    obs = env._get_current_observation()
    assert obs.shape == (1, 5, history_length)
    
    _open = 47.192501068115234 / MAX_SHARE_PRICE
    _high = 47.209999084472656 / MAX_SHARE_PRICE
    _low = 46.5525016784668 / MAX_SHARE_PRICE
    _close = 47.037498474121094 / MAX_SHARE_PRICE
    _vol = 92936000 / MAX_VOL_SHARES
    assert obs[0, 0, history_length - 1] == approx(_open)
    assert obs[0, 1, history_length - 1] == approx(_high)
    assert obs[0, 2, history_length - 1] == approx(_low)
    assert obs[0, 3, history_length - 1] == approx(_close)
    assert obs[0, 4, history_length - 1] == approx(_vol)
