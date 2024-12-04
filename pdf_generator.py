import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from fpdf import FPDF

# Step 1: Define Stocks and Portfolio Weights
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # List of stocks in the portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal portfolio allocation

# Step 2: Download Historical Data
data = yf.download(stocks, start="2010-01-01", end="2023-12-01")['Adj Close']

# Step 3: Calculate Daily Returns
returns = data.pct_change().dropna()

# Step 4: Calculate Expected Return and Risk
expected_returns = returns.mean()  # Daily expected returns
portfolio_return = np.dot(weights, expected_returns) * 252  # Annualized expected return
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized portfolio volatility

# Print Expected Return and Portfolio Volatility
print(f"Expected Annual Portfolio Return: {portfolio_return*100:.2f}%")
print(f"Portfolio Volatility (Risk): {portfolio_volatility*100:.2f}%")

# Step 5: Simulate Portfolios for Efficient Frontier
num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # To store returns, volatility, and Sharpe ratio

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    
    # Calculate portfolio returns and risk
    port_return = np.dot(weights, expected_returns) * 252  # Annualized return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized risk
    sharpe_ratio = port_return / port_volatility  # Sharpe ratio
    
    results[0,i] = port_return
    results[1,i] = port_volatility
    results[2,i] = sharpe_ratio

# Step 6: Plot Efficient Frontier
plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')

# Save the Efficient Frontier plot to a file
plot_filename = "efficient_frontier.png"
plt.savefig(plot_filename)
plt.close()

# Step 7: Generate PDF Report
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Portfolio Report", ln=True, align='C')

# Portfolio Allocation
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.cell(200, 10, txt="Portfolio Allocation", ln=True)
for stock, weight in zip(stocks, weights):
    pdf.cell(200, 10, txt=f"{stock}: {weight*100:.2f}%", ln=True)

# Expected Return and Risk Metrics
pdf.ln(10)
pdf.cell(200, 10, txt=f"Expected Annual Return: {portfolio_return*100:.2f}%", ln=True)
pdf.cell(200, 10, txt=f"Portfolio Volatility (Risk): {portfolio_volatility*100:.2f}%", ln=True)

# Insert Efficient Frontier Plot into PDF
pdf.ln(10)
pdf.cell(200, 10, txt="Efficient Frontier", ln=True)
pdf.image(plot_filename, x=10, y=pdf.get_y(), w=180)

# Output PDF
pdf.output("portfolio_report.pdf")
print("PDF report generated: portfolio_report.pdf")

# Step 8: Generate CSV Report
portfolio_data = {
    'Stock': stocks,
    'Weight': weights,
    'Expected Return (%)': expected_returns * 252 * 100,  # Annualized return
    'Portfolio Volatility (%)': portfolio_volatility * 100  # Annualized volatility
}

# Create DataFrame and save to CSV
df = pd.DataFrame(portfolio_data)
df.to_csv('portfolio_report.csv', index=False)
print("CSV report generated: portfolio_report.csv")
