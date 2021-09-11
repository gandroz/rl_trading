import random
import gym
from gym import spaces
import numpy as np


MAX_ACCOUNT_BALANCE = 1000000
MAX_VOL_SHARES = 1000000000
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super().__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(406,), dtype=np.float16)

        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        data = self.df.loc[self.current_step - 99: self.current_step, 'Open'].values / MAX_SHARE_PRICE
        data +=  self.df.loc[self.current_step - 99: self.current_step, 'High'].values / MAX_SHARE_PRICE
        data += self.df.loc[self.current_step - 99: self.current_step, 'Low'].values / MAX_SHARE_PRICE
        data += self.df.loc[self.current_step - 99: self.current_step, 'Close'].values / MAX_SHARE_PRICE
        data += self.df.loc[self.current_step - 99: self.current_step, 'Volume'].values / MAX_VOL_SHARES

        # Append additional data and scale each value to between 0-1
        data.append(self.balance / MAX_ACCOUNT_BALANCE)
        data.append(self.max_net_worth / MAX_ACCOUNT_BALANCE)
        data.append(self.shares_held / MAX_VOL_SHARES)
        data.append(self.cost_basis / MAX_SHARE_PRICE)
        data.append(self.total_shares_sold / MAX_VOL_SHARES)
        data.append(self.total_sales_value / (MAX_VOL_SHARES * MAX_SHARE_PRICE))

        return data

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Open"]
        shares = (action * self.max_stock).astype(int)

        if shares < 0:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares)
            self.shares_held += shares

        elif shares > 0:
            # Sell amount % of shares held
            self.balance += shares * current_price
            self.shares_held -= shares
            self.total_shares_sold += shares
            self.total_sales_value += shares * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')