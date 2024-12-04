import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch Historical Data
def fetch_data(symbols, start_date, end_date):
    """
    Fetch historical data for a list of symbols.
    :param symbols: List of stock or index symbols.
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    :return: DataFrame with adjusted closing prices.
    """
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    return data

# Step 2: Define Portfolio Allocation and Calculate Returns
def calculate_portfolio_returns(prices, weights):
    """
    Calculate portfolio daily and cumulative returns.
    :param prices: DataFrame of stock prices.
    :param weights: Dictionary of stock weights (e.g., {'AAPL': 0.4, 'MSFT': 0.6}).
    :return: Portfolio daily and cumulative returns.
    """
    # Normalize prices (start all stocks at $1)
    normalized_prices = prices / prices.iloc[0]
    
    # Convert weights to Series
    weight_series = pd.Series(weights)
    
    # Calculate daily portfolio value
    portfolio_value = (normalized_prices * weight_series).sum(axis=1)
    
    # Calculate daily returns
    daily_returns = portfolio_value.pct_change()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    return daily_returns, cumulative_returns

# Step 3: Backtest the Portfolio
def backtest_portfolio(portfolio_symbols, portfolio_weights, benchmark_symbol, start_date, end_date):
    """
    Backtest portfolio against a benchmark.
    :param portfolio_symbols: List of portfolio stock symbols.
    :param portfolio_weights: Dictionary of portfolio weights.
    :param benchmark_symbol: Benchmark index symbol (e.g., '^GSPC' for S&P 500).
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    """
    # Fetch portfolio and benchmark data
    symbols = portfolio_symbols + [benchmark_symbol]
    data = fetch_data(symbols, start_date, end_date)
    
    # Split data into portfolio and benchmark
    portfolio_data = data[portfolio_symbols]
    benchmark_data = data[benchmark_symbol]
    
    # Calculate portfolio returns
    portfolio_daily, portfolio_cumulative = calculate_portfolio_returns(portfolio_data, portfolio_weights)
    
    # Calculate benchmark returns
    benchmark_daily = benchmark_data.pct_change()
    benchmark_cumulative = (1 + benchmark_daily).cumprod() - 1
    
    # Evaluate performance
    sharpe_ratio = (portfolio_daily.mean() * 252) / (portfolio_daily.std() * np.sqrt(252))  # Annualized Sharpe Ratio
    max_drawdown = (portfolio_cumulative / portfolio_cumulative.cummax() - 1).min()
    
    # Display performance metrics
    print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Portfolio Maximum Drawdown: {max_drawdown:.2%}")
    
    # Plot performance
    plot_performance(portfolio_cumulative, benchmark_cumulative)

# Step 4: Plot Portfolio vs. Benchmark Performance
def plot_performance(portfolio_cumulative, benchmark_cumulative):
    """
    Plot portfolio vs benchmark cumulative returns.
    :param portfolio_cumulative: Portfolio cumulative returns.
    :param benchmark_cumulative: Benchmark cumulative returns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative, label='Portfolio', color='blue')
    plt.plot(benchmark_cumulative, label='Benchmark', color='green', linestyle='--')
    plt.title('Portfolio vs Benchmark Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()

# Step 5: Run the Backtest
if __name__ == "__main__":
    # Define portfolio stocks and weights
    portfolio_symbols = ['AAPL', 'MSFT', 'GOOG']  # Portfolio stocks
    portfolio_weights = {'AAPL': 0.4, 'MSFT': 0.4, 'GOOG': 0.2}  # Portfolio weights

    # Define benchmark
    benchmark_symbol = '^GSPC'  # S&P 500 index symbol

    # Define date range
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Run backtest
    backtest_portfolio(portfolio_symbols, portfolio_weights, benchmark_symbol, start_date, end_date)
