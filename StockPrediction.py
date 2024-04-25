import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import r2_score
import streamlit as st
from datetime import datetime
