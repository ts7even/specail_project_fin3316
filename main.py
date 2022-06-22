# Importing Libraries 
from textwrap import indent
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


# Question 1: Calculate the weekly returns of the stocks
def Returns():
    pd.options.display.float_format = '{:.2%}'.format
    tickers1 = df[['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']]
    weekly_return = tickers1.pct_change(1)
    print(weekly_return) # Might need to add the dates to the data frame


# Question 2: Regression on the first52 weeks of the stock. Store the Intercept(Alpha), Coeficient(Beta), and the standard error. 
def question2():
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    first52 = df[(df['Date'] <= '2000-12-22')]
     # Might need to make a variable to subtract first52 from df to show 1000 left. 



    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data = first52).fit()
        summary_regression = model.summary(yname="Status", xname=['Intercept "Alpha"', f'{t} Beta'], title='Regression')
        # print(summary_regression, '\n\n')

    total_fisrts_52_obaservations = first52.shape[0]
    total_overall_observations = df.shape[0]
    difference_of_observations = (total_overall_observations - total_fisrts_52_obaservations)
    print(f'Total Amount of Observations in the Regression: {total_fisrts_52_obaservations}\n'f'Total Amount of Observations Left:{difference_of_observations}\n\n')

# Question 3: Perform Rolling window regression
def rolling_regression_stats():
    first52 = df[(df['Date'] <= '2000-12-22')]

    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data=first52).fit()
        coef_and_intercept = model.params.set_axis(['Alpha', f'{t} Beta']).to_string()
        std_error = model.bse.set_axis(['Alpha STD Error', f'{t} STD Error']).to_string()
        print(coef_and_intercept)
        print(std_error, '\n\n')


# Returns()
# question2()
rolling_regression_stats()