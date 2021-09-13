import pandas as pd
from pytest import approx
from rltrading.env import StockTradingEnv, MAX_SHARE_PRICE, MAX_VOL_SHARES


def test_new_observation():
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    env = StockTradingEnv(df)
    env.current_step = 100
    obs = env._next_observation()
    assert len(obs) == 505
    
    _open = 47.0574989319 / MAX_SHARE_PRICE
    _high = 47.4124984741 / MAX_SHARE_PRICE
    _low = 46.9124984741 / MAX_SHARE_PRICE
    _close = 47.1450004578 / MAX_SHARE_PRICE
    _vol = 69844000 / MAX_VOL_SHARES
    assert obs[99] == approx(_open)
    assert obs[199] == approx(_high)
    assert obs[299] == approx(_low)
    assert obs[399] == approx(_close)
    assert obs[499] == approx(_vol)
    