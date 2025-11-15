# import necessary libraries

import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the stock symbol and download data

stock_symbol = "AAPL" # Apple Inc, as an example

data = yf.download(stock_symbol, start="2020-01-01", end="2021-12-31")

# Extract the adjusted closing price (typically used for stock predictions)
data = data[['Adj Close']]

# Create a new column for the next day's adjusted closing price (shifted -1 day)

data['Next Close'] = data['Adj Close'].shift(-1)

# Drop the last row (NaN) because there's no next day's data
data = data.dropna()

# Create features (X) and target (Y)

# X is the current day's adjusted closing price
[span_0](start_span)X = data[['Adj Close']].values #[span_0](end_span)

# Y is the next day's adjusted closing price
[span_1](start_span)Y = data[['Next Close']].values #[span_1](end_span)

# Split the data into training and testing sets

# Correction: Use the defined variables X and Y in the split function
[span_2](start_span)x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)[span_2](end_span)

# Create a linear regression model

model = LinearRegression()

# Fit the model to the training data

model.fit(x_train, y_train)

# Make predictions on the test data

y_pred = model.predict(x_test)

# Visualize the actual vs. predicted prices

plt.figure(figsize=(10,6))

# Scatter plot of actual values
plt.scatter(x_test, y_test, color='blue', label='Actual')

# Line plot of the regression model's prediction
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')

plt.xlabel('Adjusted Closing Price')

plt.ylabel("Next Day's Adjusted Closing Price")

plt.title(f'Simple Linear Regression for {stock_symbol} Stock Price Prediction')

plt.legend()

plt.grid(True) # Added grid for better readability
plt.show()
