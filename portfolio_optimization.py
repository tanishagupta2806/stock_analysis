import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ================================
# Simulated Data for Portfolio
# ================================
mean_returns = np.array([0.1, 0.12, 0.14])  # Expected annual returns for 3 assets
cov_matrix = np.array([
    [0.04, 0.02, 0.01],
    [0.02, 0.03, 0.02],
    [0.01, 0.02, 0.05]
])  # Covariance matrix of asset returns
risk_free_rate = 0.02

# Factor scores for Factor Investing (momentum, value, size)
momentum_scores = np.array([0.8, 0.5, 0.9])
value_scores = np.array([0.7, 0.8, 0.4])
size_scores = np.array([0.6, 0.9, 0.5])

# Desired sector allocation constraints
sector_allocations = np.array([0.6, 0.2, 0.2])  # Tech, Energy, Healthcare
sector_matrix = np.array([
    [1, 0, 0],  # Tech stocks
    [0, 1, 0],  # Energy stocks
    [0, 0, 1]   # Healthcare stocks
])

# ================================
# Utility Functions
# ================================

# Normalize factor scores
def normalize(scores):
    return scores / np.sum(scores)

momentum_scores = normalize(momentum_scores)
value_scores = normalize(value_scores)
size_scores = normalize(size_scores)

# Combine factor scores with weights
factor_weights = 0.4 * momentum_scores + 0.4 * value_scores + 0.2 * size_scores

# ================================
# Multi-Objective Optimization
# ================================

# Multi-objective optimization function
def multi_objective(weights, mean_returns, cov_matrix, risk_free_rate, preference_factor=0.5):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -((1 - preference_factor) * sharpe_ratio - preference_factor * portfolio_volatility)

# Factor optimization objective
def factor_objective(weights, factor_weights):
    return -np.dot(weights, factor_weights)  # Maximize factor alignment

# Constraints: Weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(mean_returns)))

# Sector allocation constraints
sector_constraints = ({'type': 'ineq', 'fun': lambda weights: sector_allocations - np.dot(sector_matrix.T, weights)})

# Combine constraints
all_constraints = [constraints, sector_constraints]

# ================================
# Optimization and Pareto Front
# ================================

# Optimize for Pareto front (multi-objective optimization)
preference_factors = np.linspace(0, 1, 10)  # Range of risk-return preferences
pareto_portfolios = []

for pf in preference_factors:
    result = minimize(multi_objective, len(mean_returns) * [1 / len(mean_returns)], args=(mean_returns, cov_matrix, risk_free_rate, pf),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    pareto_portfolios.append(result.x)

# Factor optimization
factor_result = minimize(factor_objective, len(factor_weights) * [1 / len(factor_weights)], args=(factor_weights),
                         method='SLSQP', bounds=bounds, constraints=all_constraints)

# Portfolio with all constraints
custom_result = minimize(multi_objective, len(mean_returns) * [1 / len(mean_returns)], args=(mean_returns, cov_matrix, risk_free_rate, 0.5),
                         method='SLSQP', bounds=bounds, constraints=all_constraints)

# ================================
# Visualization
# ================================

# Pareto front visualization
returns = []
volatilities = []

for weights in pareto_portfolios:
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    returns.append(portfolio_return)
    volatilities.append(portfolio_volatility)

# Plot Pareto front
plt.figure(figsize=(10, 6))
plt.plot(volatilities, returns, marker='o', label='Pareto Front (Risk vs Return)')
plt.title('Pareto Front: Risk-Return Tradeoff')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.show()

# Display results
print("Pareto Front Weights:")
for i, weights in enumerate(pareto_portfolios):
    print(f"Preference Factor {preference_factors[i]:.2f}: {weights}")

print("\nOptimized weights considering factors:", factor_result.x)
print("Portfolio weights with sector constraints:", custom_result.x)
