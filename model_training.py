import numpy as np  # Import numpy library and alias it as np
import pandas as pd  # Import pandas library and alias it as pd
import matplotlib.pyplot as plt  # Import matplotlib library for plotting and alias it as plt
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
from datetime import datetime  # Import datetime class from datetime module
"""# **Load Stock dataset from Yahoo finance**"""

start = '2010-01-01'  # Define the start date for fetching data
end = datetime.now().strftime('%Y-%m-%d')  # Define the end date as current date
df = yf.download("TSLA", start=start, end=end)  # Download Tesla (TSLA) stock data from Yahoo Finance
df.head()  # Display the first few rows of the dataset

df.tail()  # Display the last few rows of the dataset

df.describe()  # Display summary statistics of the dataset

df.drop('Adj Close',axis=1,inplace=True)  # Drop 'Adj Close' column from the DataFrame

plt.figure(figsize=(12,6))  # Create a figure with size 12x6 inches
plt.plot(df.Close)  # Plot the 'Close' column of the DataFrame
plt.show()  # Display the plot

ma50 = df.Close.rolling(50).mean()  # Calculate 50-day Moving Average (MA) of 'Close' column
plt.figure(figsize=(12,6))  # Create a new figure
plt.plot(df.Close)  # Plot the 'Close' column of the DataFrame
plt.plot(ma50, 'r')  # Plot the 50-day MA in red
plt.show()  # Display the plot

df.shape  # Display the shape of the DataFrame

"""## Splitting Data into Training & Testing"""

# Split the data into training (70%) and testing (30%) sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

print(data_training.shape)  # Print the shape of training data
print(data_testing.shape)  # Print the shape of testing data

fig, ax = plt.subplots(1, figsize=(13, 7))  # Create a new figure with one subplot
ax.plot(data_training, label='training', linewidth=2)  # Plot training data with label 'training'
ax.plot(data_testing, label='test', linewidth=2)  # Plot testing data with label 'test'
ax.set_ylabel('price USD', fontsize=14)  # Set the y-axis label
ax.legend(loc='best', fontsize=16)  # Add legend with best location and fontsize 16
plt.show()  # Display the plot

"""## Scalling of the Data"""

scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize MinMaxScaler to scale data between 0 and 1

# Scale the training data
data_training_array = scaler.fit_transform(data_training)

# Print the scaled training data
data_training_array

# Print the shape of scaled training data
data_training_array.shape

x_train = []  # Initialize empty list for input data
y_train = []  # Initialize empty list for output data
seq_len = 100  # Define sequence length

# Create input sequences and corresponding outputs for training
for i in range(seq_len, data_training_array.shape[0]):
    x_train.append(data_training_array[i-seq_len: i])  # Append input sequence
    y_train.append(data_training_array[i,0])  # Append corresponding output

x_train, y_train = np.array(x_train), np.array(y_train)  # Convert lists to numpy arrays

"""## Model"""

# Initialize a Sequential model
model = Sequential()

# Add LSTM layers with specified units, activation, and dropout
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(units=1))  # Add a Dense output layer with 1 unit

model.summary()  # Display model summary

model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model with Adam optimizer and MSE loss

model.fit(x_train, y_train, epochs=20)  # Fit the model to training data for 20 epochs

model.save('LSTM_model.h5')  # Save the trained model to a file