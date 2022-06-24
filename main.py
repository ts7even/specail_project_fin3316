# Importing Libraries 
from matplotlib import ticker
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


df = pd.read_excel("dataset\Special_Assingment_Trevor_Seibert.xlsx") 
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']

# print(df.head(5))

# Question 1: Calculate the weekly returns of the stocks
def Returns():
    pd.options.display.float_format = '{:.2%}'.format
    tickers1 = df[['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']]
    weekly_return = tickers1.pct_change(1)
    print(weekly_return) # Might need to add the dates to the data frame
    weekly_return.to_csv('source\stock_returns.csv')

# Question 2: Regression on the first52 weeks of the stock. Store the Intercept(Alpha), Coeficient(Beta), and the standard error. 
def question2():
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    first52 = df.iloc[0:52] 

    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data = first52).fit()
        summary_regression = model.summary(yname="Status", xname=['Intercept "Alpha"', f'{t} Beta'], title='Regression')
        # print(summary_regression, '\n\n')

    total_fisrts_52_obaservations = first52.shape[0]
    total_overall_observations = df.shape[0]
    difference_of_observations = (total_overall_observations - total_fisrts_52_obaservations)
    print(f'Total Amount of Observations in the Regression: {total_fisrts_52_obaservations}\n' 
    f'Total Observations in the Data {total_overall_observations}\n'
    f'Total Amount of Observations Left:{difference_of_observations}\n\n')


# Question 3: Perform Rolling window regression

def rolling_regression_stats():
    tickers = df[['FDX', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']]
    rolling_window = df
    iterable = zip(range(1110), range(52,1162))

    total_df = pd.DataFrame()
    for y, x in iterable:
        yx_df = pd.DataFrame({'Window': [f'{y}-{x}']})

        for t in tickers:
            model = smf.ols(f'{t} ~ SP50', data= rolling_window.iloc[y:x]).fit()
            beta_coef = model.params['SP50']
            std_error = model.bse['SP50']

            # window_range = (f'{y}-{x}')
            
            res = pd.DataFrame({f'{t} Beta': beta_coef, f'{t} STDERR': std_error},index=[0])
           
            yx_df = pd.concat([yx_df, res], axis=1)

        total_df = pd.concat([total_df, yx_df], axis=0, ignore_index=True)
    print(total_df)
    total_df.to_csv('source\stock_beta_stderr.csv')

Returns()
question2()    
rolling_regression_stats()

