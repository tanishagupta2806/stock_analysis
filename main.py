import numpy as np
from backtesting.backtest_portfolio import backtest_portfolio  # Ensure the import is correct

def main():
    # Example portfolio data
    stocks = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    portfolio_weights = [0.25, 0.25, 0.25, 0.25]  # Ensure this is a list or numpy array
    benchmark_symbol = 'SPY'
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    # Print the type of portfolio_weights to debug
    print(f"Type of portfolio_weights: {type(portfolio_weights)}")
    print(f"Contents of portfolio_weights: {portfolio_weights}")

    # Convert portfolio_weights to a numpy array to ensure correct format
    portfolio_weights = np.array(portfolio_weights)

    # Check again after conversion
    print(f"After conversion, type of portfolio_weights: {type(portfolio_weights)}")

    # Call the backtest function
    backtest_portfolio(stocks, portfolio_weights, benchmark_symbol, start_date, end_date)

# Run the main function
if __name__ == "__main__":
    main()
