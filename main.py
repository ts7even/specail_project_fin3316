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


pd.set_option('display.precision', 2)
# Setting up dataframe for the 10 stocks prices weekly frequency
df = pd.read_excel("dataset\Special_Assingment_Trevor_Seibert.xlsx") # This would be for a specific column

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')


# Question 1: Calculate the weekly returns of the stocks





# Question 2: Use the first 52 weeks of data points for the regression. 
# weeks = df.head(52)
# print(weeks)

def question2():
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    first52 = df[(df['Date'] <= '2000-12-22')]


    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data = first52).fit()
        print(model.summary(yname="Status", xname=['Intercept', f'{t} Beta'],  
        title='Regression'))
        print()

question2()