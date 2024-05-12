import streamlit as st  # Importing Streamlit library for creating web apps
import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
from keras.models import load_model  # Importing Keras for loading the LSTM model
import yfinance as yf  # Importing yfinance for fetching stock data
from datetime import datetime  # Importing datetime for handling dates
from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for data normalization
import matplotlib.dates as mdates  # Importing matplotlib.dates for date formatting

# Load model
model = load_model('LSTM_model.h5')  # Loading the pre-trained LSTM model

# Page configuration
st.set_page_config(page_title='Stock Prediction Dashboard', layout='wide')  # Configuring the Streamlit page layout and title
# Title and introduction
st.title('Stock Market Forecasting Dashboard')  # Adding a title to the Streamlit app
st.markdown("""
This interactive dashboard uses a Long Short-Term Memory (LSTM) network to predict stock prices based on historical data from Yahoo Finance.
Select a stock ticker, define the date range, and click the predict button to see future price projections.
""")  # Adding a markdown text for introduction

# Sidebar - User input features
st.sidebar.header('User Input Features')  # Adding a header to the sidebar
selected_stock = st.sidebar.text_input("Enter Stock Ticker", 'TSLA')  # Adding a text input for entering stock ticker in the sidebar
start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))  # Adding a date input for selecting start date in the sidebar
end_date = st.sidebar.date_input("End Date", datetime.now())  # Adding a date input for selecting end date in the sidebar

# Function to load data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)  # Downloading stock data using yfinance
    data.drop('Adj Close', axis=1, inplace=True)  # Dropping the 'Adj Close' column from the data
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initializing MinMaxScaler for normalization
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))  # Normalizing the 'Close' price data
    return scaled_data, scaler

# Function to create time steps
def create_dataset(dataset, time_step=100):
    x = []
    length = len(dataset)
    for i in range(time_step, length):
        x.append(dataset[i-time_step:i, 0])  # Creating time steps for the dataset
    return np.array(x)

# Function to plot data
def plot_data(data):
    fig, ax = plt.subplots(figsize=(15, 8))  # Creating a subplot for plotting
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')  # Plotting historical close price
    ma50 = data['Close'].rolling(50).mean()  # Calculating 50-day moving average
    ax.plot(ma50.index, ma50, label='MA 50', color='red')  # Plotting 50-day moving average
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Setting major locator for x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Setting date formatter for x-axis
    plt.title(f'Historical Close Price and Moving Average for {selected_stock}')  # Adding title to the plot
    plt.xlabel('Date')  # Adding label to x-axis
    plt.ylabel('Price (USD)')  # Adding label to y-axis
    plt.legend()  # Adding legend to the plot
    return fig  # Returning the plot figure

# Load and process data
data = load_data(selected_stock, start_date, end_date)  # Loading stock data
st.write(f"Displaying data for: {selected_stock}")  # Displaying selected stock ticker
st.pyplot(plot_data(data))  # Displaying plot of historical data

if st.button('Predict Future Prices'):  # Adding a button to trigger prediction
    scaled_data, scaler = preprocess_data(data)  # Preprocessing data for prediction
    x_test = create_dataset(scaled_data)  # Creating dataset for prediction
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)  # Reshaping data for LSTM model
    predictions = model.predict(x_test)  # Making predictions using the LSTM model
    predicted_prices = scaler.inverse_transform(predictions)  # Inverse transforming predicted prices

    # Add predictions to the plot
    future_dates = pd.date_range(start=data.index[-1], periods=len(predictions)+1, freq='B')[1:]  # Generating future dates
    fig, ax = plt.subplots(figsize=(15, 8))  # Creating a subplot for plotting
    ax.plot(data.index, data['Close'], label='Historical Close Price', color='blue')  # Plotting historical close price
    ax.plot(future_dates, predicted_prices, label='Predicted Close Price', color='green', linestyle='--')  # Plotting predicted close price
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Setting major locator for x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Setting date formatter for x-axis
    plt.title(f'Prediction of Future Close Prices for {selected_stock}')  # Adding title to the plot
    plt.xlabel('Date')  # Adding label to x-axis
    plt.ylabel('Price (USD)')  # Adding label to y-axis
    plt.legend()  # Adding legend to the plot
    st.pyplot(fig)  # Displaying plot of predicted prices

# About section
st.write("## About this Dashboard")  # Adding a header for the about section
st.info("""
This dashboard allows users to enter a stock ticker, select a date range, and view the historical closing prices and predicted future prices using a trained LSTM model. 
Data is sourced from Yahoo Finance. Predictions are based on historical data and should not be considered financial advice.
""")  # Providing information about the dashboard