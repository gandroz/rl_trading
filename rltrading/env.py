from typing import List, Union
import gym
import pandas as pd
from gym import spaces
import numpy as np



MAX_VOL_SHARES = 1000000000
MAX_SHARE_PRICE = 5000
MAX_NB_SHARES = 100
TIME_BETWEEN_BUY = 2
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000
MAX_ALLOCATION = 0.02  # 2% of the initial balance of the portfolio per transaction
MAX_PTF_ALLOCATION = 0.3  # 30% of the initial balance of the portfolio per position


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    actions_str = ["hold", "buy", "sell"]

    def __init__(self, df:pd.DataFrame, gamma:float=0.99, history_length:int=100):
        super().__init__()

        self.df = df
        self.df = self.df.sort_values('Date')
        self.gamma = gamma

        # self.action_space = spaces.Discrete(3)  # 0 to hold, 1 to buy and 2 to sell
        self.action_space = spaces.Box(low=0, 
                                       high=1,
                                       shape=(2,) ,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(1, 5, 100), dtype=np.float32)

        self.liquidity = 0
        self.net_worth = 0
        self.shares_held = 0
        self.cost_basis = 0
        self.current_step = 0
        self.value_before_buying = 0
        self.total_asset = 0
        self.cumulative_reward = 0
        self.history_length = history_length
        self.last_buy_step = 0

        # self.reset()

    def _get_current_observation(self) -> np.ndarray:
        # Get the stock data points for the last 5 days and scale to between 0-1
        history = self.df.iloc[self.current_step - self.history_length:self.current_step]
        _open = history.Open.values
        _high = history.High.values
        _low = history.Low.values
        _close = history.Close.values
        _vol = history.Volume.values

        obs = np.vstack([_open, _high, _low, _close, _vol]).reshape(1, 5, self.history_length)

        return obs

    def _apply_action(self, action:List[np.int16], obs:np.ndarray):
        current_open_price = obs[0, 0, -1]

        action_type = (action[0] * 3 - 1e-6).astype(np.int32)

        if action_type == 1:
            # Buy
            prev_cost = self.cost_basis * self.shares_held
            # do not allocate more than 2% of initial ptf value to this transaction
            # and certainly no more than the balance
            # available = min(self.value_before_buying * MAX_ALLOCATION,  # 2%
            #                 self.liquidity)
            # do not allocate more than 30% of the initial ptf value to this cumulative position
            # available = min(available, 
            #                 max(0, self.value_before_buying * MAX_PTF_ALLOCATION - prev_cost))  # 30%
            # shares = int(available / current_open_price)
            # do not open new position if previous was two days old
            # if self.current_step - self.last_buy_step < TIME_BETWEEN_BUY:
            #     shares = 0
            # else:
            #     self.last_buy_step = self.current_step
            shares = max(1, (action[1] * 1000).astype(np.int32))
            additional_cost = shares * current_open_price

            self.liquidity -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares)
            self.shares_held += shares

        elif action_type == 2:
            # Sell all shares
            self.liquidity += self.shares_held * current_open_price
            self.shares_held = 0
            self.value_before_buying = self.liquidity

        self.net_worth = self.liquidity + self.shares_held * current_open_price

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action:List[np.int16]):
        # Execute one time step within the environment
        obs = self._get_current_observation()
        self._apply_action(action, obs)

        done = self.net_worth <= 0
        
        if self.current_step > len(self.df) - 2:
            self.current_step = len(self.df) - 1
            done = True            
        
        # total_asset = self.liquidity + (self.shares_held * self.cost_basis)
        total_asset = self.liquidity + (self.shares_held * self.cost_basis)
        reward = (total_asset - self.total_asset) / (INITIAL_ACCOUNT_BALANCE)
        self.total_asset = total_asset
        self.cumulative_reward = self.cumulative_reward * self.gamma + reward

        if done:
            reward = self.cumulative_reward
        
        self.current_step += 1

        return obs, reward, done, {"action": self.actions_str[action[0].astype(np.int32)]}

    def reset(self) -> np.ndarray:
        # Reset the state of the environment to an initial state
        self.liquidity = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.value_before_buying = INITIAL_ACCOUNT_BALANCE
        self.total_asset = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.cumulative_reward = 0
        self.last_buy_step = 0
        return self._get_current_observation()

    def render(self, mode:str='human', close:bool=False, info:dict={}) -> dict:
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        return {"step": self.current_step, 
                "balance": self.liquidity,
                "shares_held": self.shares_held,
                "net_worth": self.net_worth,
                "profit": profit,
                "action": info.get("action", "unk")}