import pandas as pd
#from utils import data_string_to_float, status_calc
import gym 
from gym import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import matplotlib.pyplot as plt

#from utils import build_data_set
class StockTradingEnvironment(gym.Env):
    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()

        # Data
        self.df = df
        self.tickers = df.columns[1:]  # Assuming the first column is the date column

        # Action space: Buy, Sell, Hold for each ticker
        self.action_space = spaces.MultiDiscrete([3] * len(self.tickers))

        # Observation space: Concatenation of date, stock prices for each ticker
        num_features = 1 + len(self.tickers)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_features,), dtype=np.float32)

        # Initial state
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
        # Execute the action and move to the next time step
        self.current_step += 1
        print("action:",action )
        # Calculate reward, done, and info based on your custom logic
        reward = self._calculate_reward(action)
        done = self.current_step == self.max_steps
        info = {}

        # Update the state and get the next observation
        obs = self._next_observation()

        return obs, reward, done, info


    def _next_observation(self):
        # Extract date and prices for the current time step for each ticker
        date = datetime.strptime(self.df.iloc[self.current_step, 0], "%Y-%m-%d")
        print(date)
        days_since_start = (date - datetime.strptime(self.df.iloc[0, 0], "%Y-%m-%d")).days
        prices = self.df.iloc[self.current_step, 1:].values.astype(np.float32)
        obs = np.concatenate(([days_since_start], prices))
        return obs


    def _calculate_reward(self, action):
        # action is a list of integers representing buy, sell, or hold for each ticker
        # basic strategy where you buy one share of each stock if action is 0 (buy)
        # and sell if action is 1 (sell), and hold if action is 2 (hold)

        # Prices for the current time step
        prices = self.df.iloc[self.current_step, 1:].values.astype(np.float32)

        # Calculate the portfolio value based on the actions
        for i in range(len(self.tickers)):
            if action[i] == 0:  # Buy
                self.portfolio_value += prices[i]
            elif action[i] == 1:  # Sell
                self.portfolio_value -= prices[i]

        # Calculate the daily returns
        daily_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value if self.prev_portfolio_value != 0 else 0

        # Update the list of daily returns
        self.returns.append(daily_return)

        # Calculate the Sharpe ratio using the daily returns
        sharpe_ratio = self._calculate_sharpe_ratio(self.returns)

        # Update previous portfolio value for the next time step
        self.prev_portfolio_value = self.portfolio_value

        print("Portfolio Value:", self.portfolio_value)
        print("Previous Portfolio Value:", self.prev_portfolio_value)
        print("Daily Return:", daily_return)
        print("Sharpe Ratio:", sharpe_ratio)

        return sharpe_ratio


    def _calculate_sharpe_ratio(self, returns):
        average_return = np.mean(returns)
        risk = np.std(returns)

        epsilon = 1e-8  # Small epsilon value to avoid division by zero

        return average_return / max(risk, epsilon)


def predict_stocks_ppo(model, env):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
        return env.portfolio_value

def predict_stocks():
    # Load historical stock data
    '''
    stock_prices.csv file
    format does not have a 'ticker' column, 
    instead it has a date column, and then a column 
    for every ticker where the rows are the prices on that date
    '''
    df = pd.read_csv("stock_prices.csv")


    #any data missing, replace with either existing previous data, or future data
    df.ffill(inplace=True)
    df.bfill(inplace=True)


    


    # Create and initialize the trading environment
    env = StockTradingEnvironment(df)

    # Train the PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_stock_trading_model")

    # Load the trained model
    loaded_model = PPO.load("ppo_stock_trading_model")
    obs = env.reset()
    # Lists to store results for plotting
    years = []
    portfolio_values = []

    for _ in range(env.max_steps):
        action, _ = loaded_model.predict(obs)
        obs, _, _, _ = env.step(action)

        # Extract year from the date and store results
        year = int(env.df.iloc[env.current_step, 0][:4])
        portfolio_value = env.portfolio_value

        years.append(year)
        portfolio_values.append(portfolio_value)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(years, portfolio_values, label='Portfolio Value')
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.show()

    # Get the final portfolio value
    final_portfolio_value = env.portfolio_value
    print("Final Portfolio Value:", final_portfolio_value)

    # Get the final portfolio value
    final_portfolio_value = env.portfolio_value
    print("Final Portfolio Value:", final_portfolio_value)
    # Use the trained model for prediction

# The percentage by which a stock has to beat the S&P500 to be considered a 'buy'
OUTPERFORMANCE = 10

#Old Code from base project

# def build_data_set():
#     """
#     Reads the keystats.csv file and prepares it for scikit-learn
#     :return: X_train and y_train numpy arrays
#     """
#     training_data = pd.read_csv("keystats.csv", index_col="Date")
#     training_data.dropna(axis=0, how="any", inplace=True)
#     features = training_data.columns[6:]

#     X_train = training_data[features].values
#     # Generate the labels: '1' if a stock beats the S&P500 by more than 10%, else '0'.
#     y_train = list(
#         status_calc(
#             training_data["stock_p_change"],
#             training_data["SP500_p_change"],
#             OUTPERFORMANCE,
#         )
#     )

#     return X_train, y_train


    
# def predict_stocks():
#     X_train, y_train = build_data_set()
#     # Remove the random_state parameter to generate actual predictions
#     clf = RandomForestClassifier(n_estimators=100, random_state=0)
#     clf.fit(X_train, y_train)

#     # Now we get the actual data from which we want to generate predictions.
#     data = pd.read_csv("forward_sample.csv", index_col="Date")
#     data.dropna(axis=0, how="any", inplace=True)
#     features = data.columns[6:]
#     X_test = data[features].values
#     z = data["Ticker"].values

#     # Get the predicted tickers
#     y_pred = clf.predict(X_test)
#     if sum(y_pred) == 0:
#         print("No stocks predicted!")
#     else:
#         invest_list = z[y_pred].tolist()
#         print(
#             f"{len(invest_list)} stocks predicted to outperform the S&P500 by more than {OUTPERFORMANCE}%:"
#         )
#         print(" ".join(invest_list))
#         return invest_list


if __name__ == "__main__":
    print("Building dataset and predicting stocks...")
    predict_stocks()