# Importing Libraries 
from statistics import stdev
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
df_returns = pd.read_csv('source\stock_returns.csv')
# print(df.head(5))

# Question 1: Calculate the weekly returns of the stocks
def Returns():
    # pd.options.display.float_format = '{:.2%}'.format
    tickers1 = df[['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS', 'SP50']]
    weekly_return = tickers1.pct_change(1)
    print(weekly_return) # Might need to add the dates to the data frame
    weekly_return.to_csv('source\stock_returns.csv')

# Question 2: Regression on the first52 weeks of the stock. Store the Intercept(Alpha), Coeficient(Beta), and the standard error. 
def question2():
    tickers = ['FDX', 'BRK', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    first52 = df_returns.iloc[0:52] 

    for t in tickers:
        model = smf.ols(f'{t} ~ SP50', data = first52).fit()
        summary_regression = model.summary(yname="Status", xname=['Intercept "Alpha"', f'{t} Beta'], title='Regression')
        print(summary_regression, '\n\n')

    total_fisrts_52_obaservations = first52.shape[0]
    total_overall_observations = df.shape[0]
    difference_of_observations = (total_overall_observations - total_fisrts_52_obaservations)
    print(f'Total Amount of Observations in the Regression: {total_fisrts_52_obaservations}\n' 
    f'Total Observations in the Data {total_overall_observations}\n'
    f'Total Amount of Observations Left:{difference_of_observations}\n\n')


# Question 3: Perform Rolling window regression (Have to do it with Returns not prices)
def rolling_regression_stats():
    tickers = ['FDX', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']
    rolling_window = df_returns
    iterable = zip(range(1110), range(52,1162))

    total_df = pd.DataFrame()
    for y, x in iterable:
        yx_df = pd.DataFrame({'Window': [f'{y}-{x}']})

        for t in tickers:
            model = smf.ols(f'{t} ~ SP50', data=rolling_window.iloc[y:x]).fit()
            beta = model.params['SP50']
            alpha = model.params['Intercept']
            std_error = model.bse['Intercept']

            # window_range = (f'{y}-{x}')
            
            res = pd.DataFrame({f'{t} Beta': beta,
                                f'{t} Alpha': alpha,
                                f'{t} STDERR': std_error
                                },index=[0])
           
            yx_df = pd.concat([yx_df, res], axis=1)

        total_df = pd.concat([total_df, yx_df], axis=0, ignore_index=True)
    # print(total_df)
    total_df.to_csv('source\stock_beta_stderr.csv')



def portfolioConstruction():
    df1 = pd.read_csv('source\stock_beta_stderr.csv')
    df2 = pd.read_csv('source\stock_returns.csv')
    
    FDX_weight = (df1['FDX Alpha']/df1['FDX STDERR'])
    MSFT_weight = (df1['MSFT Alpha']/df1['MSFT STDERR'])
    NVDA_weight = (df1['NVDA Alpha']/df1['NVDA STDERR'])
    INTC_weight = (df1['INTC Alpha']/df1['INTC STDERR'])
    AMD_weight = (df1['AMD Alpha']/df1['AMD STDERR'])
    JPM_weight = (df1['JPM Alpha']/df1['JPM STDERR'])
    T_weight = (df1['T Alpha']/df1['T STDERR'])
    AAPL_weight = (df1['AAPL Alpha']/df1['AAPL STDERR'])
    AMZN_weight = (df1['AMZN Alpha']/df1['AMZN STDERR'])
    GS_weight = (df1['GS Alpha']/df1['GS STDERR'])
    # Optiomized Weight
    FDX_return = (FDX_weight*df2['FDX']).dropna()
    MSFT_return = (MSFT_weight*df2['MSFT']).dropna()
    NVDA_return = (NVDA_weight*df2['NVDA']).dropna()
    INTC_return = (INTC_weight*df2['INTC']).dropna()
    AMD_return = (AMD_weight*df2['AMD']).dropna()
    JPM_return = (JPM_weight*df2['JPM']).dropna()
    T_return = (T_weight*df2['T']).dropna()
    AAPL_return = (AAPL_weight*df2['AAPL']).dropna()
    AMZN_return = (AMZN_weight*df2['AMZN']).dropna()
    GS_return = (GS_weight*df2['GS']).dropna()
    # Equal Weight
    FDX_return1 = (0.10*df2['FDX']).dropna()
    MSFT_return1 = (0.10*df2['MSFT']).dropna()
    NVDA_return1 = (0.10*df2['NVDA']).dropna()
    INTC_return1 = (0.10*df2['INTC']).dropna()
    AMD_return1 = (0.10*df2['AMD']).dropna()
    JPM_return1 = (0.10*df2['JPM']).dropna()
    T_return1 = (0.10*df2['T']).dropna()
    AAPL_return1 = (0.10*df2['AAPL']).dropna()
    AMZN_return1 = (0.10*df2['AMZN']).dropna()
    GS_return1 = (0.10*df2['GS']).dropna()

    return_of_portfolio = (FDX_return+MSFT_return+NVDA_return+NVDA_return+INTC_return+AMD_return+JPM_return+T_return+AAPL_return+AMZN_return+GS_return)
    equal_weight_return = (FDX_return1+MSFT_return1+NVDA_return1+NVDA_return1+INTC_return1+AMD_return1+JPM_return1+T_return1+AAPL_return1+AMZN_return1+GS_return1)
    std_dev_opt_port = stdev(return_of_portfolio)
    std_dev_equal_port = stdev(equal_weight_return)
    cumul_return_portfolio = return_of_portfolio.pct_change()
    # print(return_of_portfolio)
    # print(equal_weight_return)
    # print(std_dev_opt_port)
    # print(std_dev_equal_port)








# Returns()
# question2()    
# rolling_regression_stats()
# portfolioConstruction()
