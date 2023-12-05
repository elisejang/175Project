import pandas as pd
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from datetime import datetime

from stock_prediction import StockTradingEnvironment
class BuyHoldStockTradingEnvironment(gym.Env):
    def __init__(self, df):
        super(BuyHoldStockTradingEnvironment, self).__init__()

        # Data
        self.df = df
        self.tickers = df.columns[1:]  # The first column is the date column
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Portfolio variables
        self.portfolio_value = 0
        self.prev_portfolio_value = 0
        self.returns = []

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 0
        self.prev_portfolio_value = 0
        self.returns = []
        return self._next_observation()

    def step(self, action):
        # For a buy and hold strategy, the action is not used
        # Execute the action and move to the next time step
        self.current_step += 1

        # Calculate reward, done, and info based on your custom logic
        reward = self._calculate_reward()
        done = self.current_step == self.max_steps
        info = {}

        # Update the state and get the next observation
        obs = self._next_observation()

        return obs, reward, done, info

    def _next_observation(self):
        # Extract date and prices for the current time step for each ticker
        date = self.df.iloc[self.current_step, 0]
        prices = self.df.iloc[self.current_step, 1:].values.astype(np.float32)

        # Return a single value for observation
        obs = np.concatenate(([self.current_step], prices))

        return obs

    def _calculate_reward(self):
        # Prices for the current time step
        prices = self.df.iloc[self.current_step, 1:].values.astype(np.float32)

        # Calculate the portfolio value based on the buy and hold strategy
        if self.current_step == 0:
            # Buy at the beginning of the period
            self.portfolio_value = np.sum(prices)
        else:
            # Hold throughout the period
            self.portfolio_value = np.sum(prices)

        # Calculate the daily returns
        daily_return = (self.portfolio_value - self.prev_portfolio_value) / abs(self.prev_portfolio_value) if abs(
            self.prev_portfolio_value) > 0 else 0
        # Update the list of daily returns
        self.returns.append(daily_return)

        # Update previous portfolio value for the next time step
        self.prev_portfolio_value = self.portfolio_value

        print("Portfolio Value (Buy and Hold):", self.portfolio_value)
        print("Daily Return (Buy and Hold):", daily_return)

        return daily_return


def buy_hold_and_ppo_predict_stocks():
    # Load historical stock data
    df = pd.read_csv("stock_train.csv")
    tf = pd.read_csv("stock_test.csv")

    # Any data missing, replace with either existing previous data, or future data
    # However, this leads to inaccuracies but whatever
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    tf.ffill(inplace=True)
    tf.bfill(inplace=True)

    # Create and initialize the Buy and Hold environment
    buy_hold_env = BuyHoldStockTradingEnvironment(tf)

    # Lists to store results for plotting
    dates = []
    buy_hold_portfolio_values = []
    ppo_portfolio_values = []

    # Run Buy and Hold strategy
    obs=buy_hold_env.reset()
    for _ in range(buy_hold_env.max_steps):
        action = None  # For buy and hold, action is not used
        obs, _, _, _ = buy_hold_env.step(action)

        date_str = buy_hold_env.df.iloc[buy_hold_env.current_step, 0]
        date = datetime.strptime(date_str, "%Y-%m-%d")

        dates.append(date)
        buy_hold_portfolio_values.append(buy_hold_env.portfolio_value)

    # Load the trained PPO model
    ppo_model = PPO.load("ppo_stock_trading_model")

    # Create and initialize the PPO environment
    ppo_env = StockTradingEnvironment(tf)

    # Run PPO strategy
    obs = ppo_env.reset()
    for _ in range(ppo_env.max_steps):
        action, _ = ppo_model.predict(obs)
        obs, _, _, _ = ppo_env.step(action)

        date_str = ppo_env.df.iloc[ppo_env.current_step, 0]
        date = datetime.strptime(date_str, "%Y-%m-%d")

        ppo_portfolio_values.append(ppo_env.portfolio_value)

    # Plotting the results
    plt.figure(figsize=(15, 8))

    # Plot with the comparison of Buy and Hold and PPO strategies
    plt.subplot(2, 1, 1)
    plt.plot(dates, buy_hold_portfolio_values, label='Portfolio Value (Buy and Hold)')
    plt.plot(dates, ppo_portfolio_values, label='Portfolio Value (PPO)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time (Buy and Hold vs PPO)')
    plt.legend()

    # Plot with only Buy and Hold strategy
    plt.subplot(2, 1, 2)
    plt.plot(dates, buy_hold_portfolio_values, label='Portfolio Value (Buy and Hold)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time (Buy and Hold Only)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    final_buy_hold_portfolio_value = buy_hold_env.portfolio_value
    final_ppo_portfolio_value = ppo_env.portfolio_value

    print("Final Portfolio Value (Buy and Hold):", final_buy_hold_portfolio_value)
    print("Final Portfolio Value (PPO):", final_ppo_portfolio_value)

if __name__ == "__main__":
    print("Building dataset and running buy and hold and PPO strategies...")
    buy_hold_and_ppo_predict_stocks()
