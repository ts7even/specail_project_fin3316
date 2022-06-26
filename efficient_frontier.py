from statistics import stdev
from matplotlib import ticker
import pandas as pd
import numpy as np 
import scipy as sp
from scipy.optimize import minimize, Bounds, LinearConstraint
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime

# Dataframe of the prices
df = pd.read_excel("dataset\Special_Assingment_Trevor_Seibert.xlsx") 
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')


# Dataframe for the Returns and the Beta, STD Error, and Alpha
df1 = pd.read_csv('source\stock_beta_stderr.csv')
df2 = pd.read_csv('source\stock_returns.csv')
# Correlation Matirx Dataset 
df3 = pd.read_csv('source\correlation_matrix.csv')


# Correlation Matrix
def correlationMatrix():
    correlation_returns = df2[['FDX', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS', 'SP50']]
    cMatrix = correlation_returns.corr()
    mask1 = np.triu(cMatrix)
    cMatrix.to_csv('source\correlation_matrix.csv')
    print(cMatrix)

    sns.heatmap(cMatrix, annot=True, annot_kws={"size":10}, mask=mask1, cmap='BuPu')
    plt.title("Heatmap of Return Correllation")
    plt.yticks(rotation=0)
    plt.show(block=False)
    plt.savefig('graphs\correlation_matrix_of_returns.png')
    plt.pause(10)
    plt.close()


df3 = df2[['FDX', 'MSFT', 'NVDA', 'INTC', 'AMD', 'JPM', 'T', 'AAPL', 'AMZN', 'GS']]
rf = 0.031339 # 10-Yr US Yeild 6/25/2022
cds = 0.002605 # 10-yr Treasury CDS 6/25/2022
rfr = (rf-cds) # True Riskfree Rate
W = np.random.uniform(0.05,0.15,size=10)
cov = df3.iloc[1058:1152].cov()*52 #Past Two Years
'''Also, need to learn Java to create a real montecarlo simulation to find the true expected return. Also, Alpha + Expected Return (Look comment below) = Actual Expected Return'''
expected_returns = W # To find a true expected return we need to do a probability distribution (returnA*probabilityA) +... 

# Optimal Weight from Dr. Estelaei 𝑤(𝑖) =  α(i)/ 𝜎2𝑒(𝑖) (Last Part is the Standard Deviation of the Residuals.)

def optimumPortfolio(func, W, exp_ret, cov, target_return):
    opt_bounds = Bounds(0,1)

    opt_constrsaints = ({'type': 'eq',
    'fun': lambda W: 1.0 -np.sum(W)},
    {'type': 'eq',
    'fun': lambda W: target_return - W.T@exp_ret})

    optimal_weights = minimize(func, W,
    args=(exp_ret, cov),
    method='SLSQP',
    bounds=opt_bounds,
    constraints=opt_constrsaints)

    return optimal_weights['x']

# Function to optimize
def ret_risk(W, exp_ret, cov):
    return - ((W.T@exp_ret) / (W.T@cov@W)**0.5)

x = optimumPortfolio(ret_risk, W, expected_returns, cov, target_return=0.0555)
print(x)