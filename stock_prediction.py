import pandas as pd
#from utils import data_string_to_float, status_calc
import gym 
from gym import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

#from utils import build_data_set
class StockTradingEnvironment(gym.Env):
    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()

        # Data
        self.df = df
        self.tickers = df.columns[1:]  # Assuming the first column is the date column

        # Action space: Buy, Sell, Hold for each ticker
        self.action_space = spaces.MultiDiscrete([3] * len(self.tickers))

        # Calculate observation space size based on the number of features
        num_features = 1 + len(self.tickers) + 7  # 1 for days_since_start, len(self.tickers) for stock prices, 4 for additional features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_features,), dtype=np.float32)

        # Initial state
        self.current_step = 0
        self.max_steps = len(df) - 1

        # Portfolio variables
        self.portfolio_value = 0
        self.prev_portfolio_value = 0
        self.returns = []
        
        #keeps track of historical ticker prices
        self.historical_prices = []
    def reset(self):
        self.current_step = 0
        self.portfolio_value = 0
        self.prev_portfolio_value = 0
        self.returns = []
        return self._next_observation()

    def step(self, action):
        # Execute the action and move to the next time step
        self.current_step += 1
        # Calculate reward, done, and info based on your custom logic
        reward = self._calculate_reward(action)
        done = self.current_step == self.max_steps
        info = {}

        # Update the state and get the next observation
        obs = self._next_observation()

        return obs, reward, done, info

    #helper function to append prices to historical prices
    def _add_to_historical_prices(self, prices ):
        if len(self.historical_prices)==20:
            self.historical_prices.pop(0)
        self.historical_prices.append(prices)
    
    #returns the mean of the window we want to look at for historical prices, prices is a 2d array
    def _mean_of_historical_prices(self, prices , window=20):
          mean = np.mean(prices[-window:], axis=0)
          return mean

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        # Calculate the mean of historical prices
        historical_prices_mean = self._mean_of_historical_prices(prices, window)
        # Calculate the standard deviation using historical prices
        historical_prices_std = np.std(historical_prices_mean)
        
        # Calculate upper and lower Bollinger Bands
        upper_band = historical_prices_mean.mean() + num_std * historical_prices_std
        lower_band = historical_prices_mean.mean() - num_std * historical_prices_std
        # if math.isnan(upper_band)  or math.isnan(lower_band) ==None:
        #     return 0, 0
        return upper_band, lower_band


    def calculate_macd(self, prices, short_window=10, long_window=20, signal_window=6):
        # Convert prices to a DataFrame
        df_prices = pd.DataFrame(prices)

        # Calculate short-term EMA
        short_ema = df_prices.ewm(span=short_window, adjust=False).mean().iloc[-1]

        # Calculate long-term EMA
        long_ema = df_prices.ewm(span=long_window, adjust=False).mean().iloc[-1]

        # Calculate MACD line
        macd_line = short_ema - long_ema

        # Calculate Signal line (9-day EMA of MACD)
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        return macd_line.values, signal_line.values
    
    def _next_observation(self):
        # Extract date and prices for the current time step for each ticker
        date = datetime.strptime(self.df.iloc[self.current_step, 0], "%Y-%m-%d")
        print(date)
        days_since_start = (date - datetime.strptime(self.df.iloc[0, 0], "%Y-%m-%d")).days
        prices = self.df.iloc[self.current_step, 1:].values.astype(np.float32)
        self._add_to_historical_prices(prices)
        # Feature engineering: Calculate moving averages, RSI, and MACD
        # Moving Averages
        short_window = 5
        long_window = 20
        short_ma = self._mean_of_historical_prices(self.historical_prices, window=short_window).mean()
        long_ma = self._mean_of_historical_prices(self.historical_prices, window=long_window).mean()

        # Relative Strength Index (RSI)
        changes = np.diff(self.historical_prices)
        print(changes)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-short_window:])
        avg_loss = np.mean(losses[-short_window:])

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        macd_line, signal_line = self.calculate_macd(self.historical_prices)

        # Return a single value for observation
        macd = macd_line[-1] - signal_line[-1]

        


        # Calculate historical volatility
        historical_prices_mean = self._mean_of_historical_prices(self.historical_prices)
        returns = (prices - historical_prices_mean) / historical_prices_mean
        volatility = np.std(returns) * np.sqrt(252)


        
        # Bollinger Bands
        upper_band, lower_band = self._calculate_bollinger_bands(self.historical_prices, window=20, num_std=2)


        print("Short ma:", short_ma)
        print("Long ma:", long_ma)
        print("RSI:", rsi)
        print("MACD:", macd)
        print("volatility:", volatility)
        print("upper band:", upper_band)
        print("lower band:", lower_band)
        
        obs = np.concatenate(([days_since_start], prices, [short_ma, long_ma, rsi, macd, volatility, upper_band, lower_band]))


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

         # Apply penalty for negative portfolio value
        penalty = 0
        if self.portfolio_value < 0:
            penalty = -1

         # Calculate the daily returns
        daily_return = daily_return = (self.portfolio_value - self.prev_portfolio_value) / abs(self.prev_portfolio_value) if abs(self.prev_portfolio_value) > 0 else 0
        # Update the list of daily returns
        self.returns.append(daily_return)

        # Calculate the trend over a specified period (e.g., one month)
        trend_window = 20  # Adjust this window size based on preference
        recent_returns = self.returns[-trend_window:]

        # Penalize if the overall trend is downwards
        if np.mean(recent_returns) < 0:
            penalty += -1  # Adjust the penalty based on preference


        # Update previous portfolio value for the next time step
        self.prev_portfolio_value = self.portfolio_value

        print("Portfolio Value:", self.portfolio_value)
        print("Previous Portfolio Value:", self.prev_portfolio_value)
        print("Daily Return:", daily_return)
        # print("Sharpe Ratio:", sharpe_ratio)

        return daily_return + penalty

    # def _calculate_sharpe_ratio(self, returns):
    #     average_return = np.mean(returns)
    #     risk = np.std(returns)

    #     epsilon = 1e-8  # Small epsilon value to avoid division by zero

    #     return average_return / max(risk, epsilon)


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
    df = pd.read_csv("stock_train.csv")
    tf = pd.read_csv("stock_test.csv")

    #any data missing, replace with either existing previous data, or future data
    #however this leads to inacurracies but whatever
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    tf.ffill(inplace=True)
    tf.bfill(inplace=True)



    # Create and initialize the trading environment
    env = StockTradingEnvironment(df)
    
    
    # Train the PPO model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_stock_trading_model")
    
    # Create and initialize the training environment
    test_env = StockTradingEnvironment(tf)

    # Load the trained model
    loaded_model = PPO.load("ppo_stock_trading_model")
    obs = test_env.reset()
    # Lists to store results for plotting
    dates = []
    portfolio_values = []


    #simulate trained model on environment
    # Use the trained model for prediction

    for _ in range(test_env.max_steps):
        action, _ = loaded_model.predict(obs)
        obs, _, _, _ = test_env.step(action)

        # Extract date and store results
        date_str = test_env.df.iloc[test_env.current_step, 0]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        portfolio_value = test_env.portfolio_value

        dates.append(date)
        portfolio_values.append(portfolio_value)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(dates, portfolio_values, label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    # Format y-axis as integers
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.show()


    # Get the final portfolio value
    final_portfolio_value = env.portfolio_value
    print("Final Portfolio Value:", final_portfolio_value)


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