import random
import gym
from gym import spaces
import numpy as np


MAX_ACCOUNT_BALANCE = 1000000
MAX_VOL_SHARES = 1000000000
MAX_SHARE_PRICE = 5000
MAX_NB_SHARES = 100
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000
MAX_ALLOCATION = 0.02  # 2% of the initial balance of the portfolio per transaction
MAX_PTF_ALLOCATION = 0.3  # 30% of the initial balance of the portfolio per position


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    actions_str = ["hold", "buy", "sell"]

    def __init__(self, df, gamma=0.99):
        super().__init__()

        self.df = df
        self.gamma = gamma

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Discrete(3)  # 0 to hold, 1 to buy and 2 to sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(505,), dtype=np.float16)
        self.balance = 0
        self.net_worth = 0
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.value_before_buying = 0
        self.total_asset = 0
        self.gamma_reward = 0
        self.reset()

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        _open = self.df.loc[self.current_step - 99: self.current_step, 'Open'].values / MAX_SHARE_PRICE
        _high =  self.df.loc[self.current_step - 99: self.current_step, 'High'].values / MAX_SHARE_PRICE
        _low = self.df.loc[self.current_step - 99: self.current_step, 'Low'].values / MAX_SHARE_PRICE
        _close = self.df.loc[self.current_step - 99: self.current_step, 'Close'].values / MAX_SHARE_PRICE
        _vol = self.df.loc[self.current_step - 99: self.current_step, 'Volume'].values / MAX_VOL_SHARES

        # Append additional data and scale each value to between 0-1
        data = np.array([self.balance / MAX_ACCOUNT_BALANCE,
                         self.shares_held / MAX_VOL_SHARES,
                         self.cost_basis / MAX_SHARE_PRICE,
                         self.total_shares_sold / MAX_VOL_SHARES,
                         self.total_sales_value / (MAX_VOL_SHARES * MAX_SHARE_PRICE)])
        obs = np.concatenate([_open, _high, _low, _close, _vol, data])
        return obs

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Open"]

        if action == 1:
            # Buy
            prev_cost = self.cost_basis * self.shares_held
            available = min(self.value_before_buying * MAX_ALLOCATION,  # 2%
                            self.balance)
            available = min(available, 
                            max(0, self.value_before_buying * MAX_PTF_ALLOCATION - prev_cost))  # 30%
            shares = int(available / current_price)            
            additional_cost = shares * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares)
            self.shares_held += shares

        elif action == 2:
            # Sell all shares
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            self.total_shares_sold += self.shares_held
            self.total_sales_value += self.shares_held * current_price
            self.value_before_buying = self.balance

        self.net_worth = self.balance + self.shares_held * current_price

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        done = self.net_worth <= 0
        
        if self.current_step > len(self.df) - 2:
            self.current_step = len(self.df) - 1
            done = True            
        
        obs = self._next_observation()

        total_asset = self.balance + (self.shares_held * self.cost_basis)
        reward = (total_asset - self.total_asset) / (INITIAL_ACCOUNT_BALANCE * MAX_ALLOCATION)
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        if done:
            reward = self.gamma_reward
        
        return obs, reward, done, {"action": self.actions_str[action]}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.value_before_buying = INITIAL_ACCOUNT_BALANCE
        self.total_asset = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.gamma_reward = 0
        return self._next_observation()

    def render(self, mode='human', close=False, info={}):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        # print(f'Step: {self.current_step}')
        # print(f'Balance: {self.balance}')
        # print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        # print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        # print(f'Net worth: {self.net_worth}')
        # print(f'Profit: {profit}')
        return {"step": self.current_step, 
                "balance": self.balance,
                "shares_held": self.shares_held,
                "net_worth": self.net_worth,
                "profit": profit,
                "action": info.get("action", "unk")}