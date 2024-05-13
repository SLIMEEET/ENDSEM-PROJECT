import numpy as np  # Import NumPy library and alias it as np
import pandas as pd  # Import Pandas library and alias it as pd
import matplotlib.pyplot as plt  # Import Matplotlib library and alias it as plt
import pandas_datareader as data  # Import pandas_datareader for fetching financial data
import yfinance as yf  # Import yfinance for accessing Yahoo Finance API
import datetime  # Import datetime module
import seaborn as sns  # Import seaborn for statistical data visualization
import statsmodels.api as sm  # Import statsmodels for statistical models and tests
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler from scikit-learn
from keras.layers import Dense, Dropout, LSTM  # type: ignore # Import Dense, Dropout, LSTM layers from Keras
from keras.models import Sequential  # type: ignore # Import Sequential model from Keras
from keras.models import load_model  # type: ignore # Import load_model function from Keras
from sklearn.metrics import r2_score  # Import r2_score from scikit-learn
import streamlit as st  # Import Streamlit library for creating web applications
from datetime import datetime  # Import datetime class from datetime module

"""# **Load Stock dataset from Yahoo finance**"""

# Define the start and end dates for the data
start = '2010-01-01'  # Define the start date for fetching data
end = datetime.now().strftime('%Y-%m-%d')  # Define the end date as the current date

# Download Tesla (TSLA) stock data from Yahoo Finance
df = yf.download("TSLA", start=start, end=end)

# Describe data
st.subheader("Data from 2010 - 2024")  # Display subheader for the data
st.write(df.head())  # Display the first few rows of the dataset
st.write(df.describe())  # Display basic statistics of the dataset

# Visualizations
st.subheader("Closing Price Vs Time Chart with 50MA")  # Display subheader for the visualization

# Calculate the 50-day Moving Average (MA) of the closing price
ma50 = df.Close.rolling(50).mean()

# Create a plot for closing price and 50-day MA
fig = plt.figure(figsize=(12, 6))  # Create a new figure with a specified size
plt.plot(ma50, 'r')  # Plot the 50-day MA in red
plt.plot(df.Close, 'b')  # Plot the closing price in blue
plt.xlabel('Date', fontsize=14)  # Set the x-axis label with fontsize
plt.ylabel('Price USD', fontsize=14)  # Set the y-axis label with fontsize
st.pyplot(fig)  # Display the plot in the Streamlit app

"""## Splitting Data into Training & Testing"""

# Split the data into training (70%) and testing (30%) sets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# Display the training and testing data
st.subheader("Training Vs Test Data")  # Display subheader for the data
fig, ax = plt.subplots(1, figsize=(12, 6))  # Create a new figure with one subplot
ax.plot(data_training, label='training', linewidth=2)  # Plot training data with label 'training'
ax.plot(data_testing, label='test', linewidth=2)  # Plot testing data with label 'test'
ax.set_xlabel('Date', fontsize=14)  # Set the x-axis label with fontsize
ax.set_ylabel('Price USD', fontsize=14)  # Set the y-axis label with fontsize
ax.set_title('', fontsize=16)  # Set the title of the plot with fontsize
ax.legend(loc='best', fontsize=16)  # Add legend with best location and fontsize 16
st.pyplot(fig)  # Display the plot in the Streamlit app

"""## Scaling of the Data"""

# Initialize the MinMaxScaler to scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the training data
data_training_array = scaler.fit_transform(data_training)

# Prepare the training data for LSTM model
x_train = []
y_train = []
seq_len = 100  # Sequence length for LSTM

# Create sequences of length seq_len for training
for i in range(seq_len, data_training_array.shape[0]):
    x_train.append(data_training_array[i - seq_len: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

"""## Model"""

# Load a pre-trained LSTM model
model = load_model('LSTM_model.h5')

# Scale the testing data
input_data = scaler.fit_transform(data_testing)

# Prepare the testing data for LSTM model
x_test = []
y_test = []

# Create sequences of length seq_len for testing
for i in range(seq_len, input_data.shape[0]):
    x_test.append(input_data[i - seq_len: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

"""## Predictions"""

# Make predictions using the LSTM model
y_pred = model.predict(x_test)

# Scale back the predictions and actual values to the original scale
scale_factor = 1 / scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

pred = y_pred.ravel()  # Flatten the predictions
actual = data_testing[seq_len:]  # Actual values after sequence length
preds = pd.Series(index=actual.index, data=pred)  # Create a pandas Series for predictions

st.subheader("Predictions Vs Original")  # Display subheader for the predictions vs. original data
fig, ax = plt.subplots(1, figsize=(12, 6))  # Create a new figure with one subplot
ax.plot(actual, label='actual', linewidth=2)  # Plot actual values with label 'actual'
ax.plot(preds, label='prediction', linewidth=2)  # Plot predictions with label 'prediction'
ax.set_xlabel('Date', fontsize=14)  # Set the x-axis label with fontsize
ax.set_ylabel('Price USD', fontsize=14)  # Set the y-axis label with fontsize
ax.set_title('', fontsize=16)  # Set the title of the plot with fontsize
ax.legend(loc='best', fontsize=16)  # Add legend with best location and fontsize 16
st.pyplot(fig)  # Display the plot in the Streamlit app

"""# Forecasting"""

# Allow the user to select a time period for future forecasting
look_back = st.slider("Select time period for Future forecasting", min_value=1, max_value=len(x_test), value=365)

# Use the last look_back days of historical data as input for forecasting
current_input = x_test[-look_back:, :, :]

# Make forecasts using the LSTM model
forecast = model.predict(current_input)
forecast = forecast * scale_factor
forecasts = forecast.ravel()  # Flatten the forecasted values

# Get the dates for original data and forecast
dates = data_testing.index[-len(y_test):]
forecast_dates = pd.date_range(start=dates[-1], periods=len(forecast) + 1, freq='D')[1:]

# Create a pandas Series for the forecasted values
forecast = pd.Series(index=forecast_dates, data=forecasts)

st.subheader("Forecasting Graph")  # Display subheader for the forecasting graph
fig, ax = plt.subplots(1, figsize=(12, 6))  # Create a new figure with one subplot
ax.plot(actual, label='actual', linewidth=2)  # Plot actual values with label 'actual'
ax.plot(forecast[1:], label='forecast', linewidth=2)  # Plot forecasted values with label 'forecast'
ax.set_xlabel('Date', fontsize=14)  # Set the x-axis label with fontsize
ax.set_ylabel('Price USD', fontsize=14)  # Set the y-axis label with fontsize
ax.set_title('', fontsize=16)  # Set the title of the plot with fontsize
ax.legend(loc='best', fontsize=16)  # Add legend with best location and fontsize 16
st.pyplot(fig)  # Display the plot in the Streamlit app