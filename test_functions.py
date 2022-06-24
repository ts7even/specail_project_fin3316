# Importing Libraries 
import pandas as pd
import numpy as np 
import scipy as sp 
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime

pd.set_option('display.precision', 2)

df = pd.read_excel("dataset\Special_Assingment_Trevor_Seibert.xlsx") 
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']

def rolling_reg():
    model = smf.ols(f'FDX ~ SP50', data=df).fit()
    coef_and_intercept = model.params['SP50'] # Beta
    std_error = model.bse['SP50'] # Beta Std
    # print(coef_and_intercept)
    # print(std_error)
    
    df1 = pd.DataFrame({
        'FDX Beta': coef_and_intercept,
        'FDX STD ERR':std_error
    },index=[0])
    print(df1)
rolling_reg()