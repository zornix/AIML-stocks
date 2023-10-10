import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set API key (please use your own)
API_KEY = 'bqPCHMpoyJBfZ3FAmtj23bq821Kx6IGh'

# Prompt the user to enter a stock ticker, start date, end date
ticker = input("Enter the company ticker: ")
start_date = input("Enter the start date for the training data (yyyy-mm-dd): ")
end_date = input("Enter the end date for the training data(yyyy-mm-dd): ")

# Get the company's actual name from the API
company_info_url = f'https://api.polygon.io/v1/meta/symbols/{ticker}/company?apiKey={API_KEY}'
response = requests.get(company_info_url)
company_name = response.json()['name']

# Construct the API request URL
url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}'

# Send the API request and parse the response
response = requests.get(url)
data = response.json()['results']

# Convert the response to a pandas dataframe
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
df = df.set_index('timestamp')
df = df.drop(columns=['t', 'v', 'vw'])

# Split the data into training and testing sets
train_data = df
test_data = train_data.iloc[-365:]

# Train a linear regression model
model = LinearRegression()
model.fit(train_data[['o', 'h', 'l', 'c']], train_data['o'])

# Use the model to predict the stock prices for the next 365 days
next_days = pd.date_range(start=test_data.index[-1], periods=365, freq='D')
next_days_df = pd.DataFrame(index=next_days, columns=train_data.columns)
next_days_df['o'] = model.predict(test_data[['o', 'h', 'l', 'c']])

# Combine the test data and the predicted data
combined_data = pd.concat([test_data, next_days_df])

# Plot the actual and predicted stock prices
plt.plot(combined_data.index, combined_data['o'], label='Actual/Predicted')
plt.legend()
plt.title(f'{company_name} ({ticker}) Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

