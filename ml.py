import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hmmlearn.hmm import GaussianHMM

# Step 1: Collect Historical Stock Data
ticker = 'AAPL'  # Example stock symbol (can be replaced with any stock symbol)
data = yf.download(ticker, start='2010-01-01', end='2024-01-01')

# Step 2: Preprocess the Data
# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()

# Calculate rolling volatility (standard deviation over 20 days)
data['Volatility'] = data['Daily Return'].rolling(window=20).std()

# Remove NaN values
data.dropna(inplace=True)

# Step 3: Prepare Data for Machine Learning
# Create a new column for next day's return (target)
data['Next Return'] = data['Daily Return'].shift(-1)
data.dropna(inplace=True)

X = data[['Daily Return', 'Volatility']]  # Features
y = data['Next Return']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Random Forest Model for Predicting Stock Returns
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize Actual vs Predicted Returns
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Returns')
plt.plot(y_test.index, y_pred, label='Predicted Returns')
plt.legend()
plt.title(f'{ticker} - Actual vs Predicted Returns')
plt.show()

# Step 5: Regime Switching Using Clustering (K-Means)
# We will use daily returns and volatility to identify market regimes
X_regimes = data[['Daily Return', 'Volatility']]

# Apply K-Means clustering to identify regimes (Bull vs Bear market)
kmeans = KMeans(n_clusters=2, random_state=42)
data['Regime'] = kmeans.fit_predict(X_regimes)

# Visualize the Clusters (Regimes)
plt.scatter(data['Daily Return'], data['Volatility'], c=data['Regime'], cmap='coolwarm')
plt.title("Market Regimes (Bull vs Bear) - KMeans")
plt.xlabel('Daily Return')
plt.ylabel('Volatility')
plt.show()

# Step 6: Regime Switching Using Hidden Markov Models (HMM)
# Apply HMM for market regime identification
X_hmm = data[['Daily Return', 'Volatility']].values

# Initialize and train the Hidden Markov Model
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
hmm_model.fit(X_hmm)

# Predict the hidden states (market regimes)
data['Regime_HMM'] = hmm_model.predict(X_hmm)

# Visualize the regimes from HMM
plt.scatter(data.index, data['Daily Return'], c=data['Regime_HMM'], cmap='coolwarm')
plt.title("Market Regimes (HMM) - Hidden Markov Model")
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()

# Step 7: Portfolio Adjustment Based on Regime
# Example portfolio: 4 assets with equal allocation initially
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
portfolio = {asset: 0.25 for asset in assets}

# Adjust portfolio based on regime
for i, regime in enumerate(data['Regime_HMM']):
    if regime == 0:  # Bull market
        portfolio['AAPL'] += 0.05  # Increase allocation to AAPL
        portfolio['GOOGL'] += 0.05
    else:  # Bear market
        portfolio['AMZN'] -= 0.05  # Decrease allocation to riskier assets like AMZN
        portfolio['MSFT'] -= 0.05

print("Adjusted Portfolio Allocation:", portfolio)

# Step 8: Predict Volatility Using Rolling Window Model (Optional - for more advanced models)
# Calculate rolling standard deviation for volatility prediction (if needed)
data['Rolling Volatility'] = data['Daily Return'].rolling(window=20).std()

# You can use this volatility data in conjunction with stock return predictions
