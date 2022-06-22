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

print(df.head(5))

# Question 1: Calculate the weekly returns of the stocks
def calculateReturns():
    print("ehllow asdo")



# Question 2: Use the first 52 weeks of data points for the regression. 
# weeks = df.head(52)
# print(weeks)

def question2():
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    first52 = df[(df['Date'] <= '2000-12-22')]
    print(first52.shape[0], '\n\n') # Might need to make a variable to subtract first52 from df to show 1000 left. 



    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data = first52).fit()
        print(model.summary(yname="Status", xname=['Intercept', f'{t} Beta'],  
        title='Regression'))
        print()



def rolling_reg(df):
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']

    for y in range(2001, 2022):  # 2001 through 2021
        df_for_one_year = df[df.Date.dt.year == y]
    
        for t in tickers:
            model = smf.ols(f'{t} ~ SP50', data=df_for_one_year).fit()
            print(model.summary(yname="Status", xname=['Intercept', f'{t} Beta'],  
            title=f'Regression {y}'))
            print()

# calculateReturns()
# question2()
rolling_reg(df)