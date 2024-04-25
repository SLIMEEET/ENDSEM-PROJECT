import numpy as np  # Import numpy library and alias it as np
import pandas as pd  # Import pandas library and alias it as pd
import matplotlib.pyplot as plt  # Import matplotlib library for plotting and alias it as plt
import pandas_datareader as data  # Import pandas_datareader for fetching financial data
import yfinance as yf  # Import yfinance for accessing Yahoo Finance API
import datetime  # Import datetime module
import seaborn as sns  # Import seaborn for statistical data visualization
import statsmodels.api as sm  # Import statsmodels for statistical models and tests
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler from scikit-learn
from keras.layers import Dense, Dropout, LSTM  # Import Dense, Dropout, LSTM layers from Keras
from keras.models import Sequential  # Import Sequential model from Keras
from keras.models import load_model  # Import load_model function from Keras
from sklearn.metrics import r2_score  # Import r2_score from scikit-learn
from datetime import datetime  # Import datetime class from datetime module