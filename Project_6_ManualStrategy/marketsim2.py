"""MC2-P1: Market simulator.
orders.csv: https://youtu.be/1ysZptg2Ypk?t=3m48s
df_prices: https://youtu.be/1ysZptg2Ypk?t=6m30s
df_trades: https://youtu.be/1ysZptg2Ypk?t=9m9s
df_holdings: https://youtu.be/1ysZptg2Ypk?t=15m47s

df_trades: A data frame whose values represent trades for each day. Legal values are +1000.0 indicating a BUY of 1000 
shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are 
also legal so long as net holdings are constrained to -1000, 0, and 1000.
"""

import pandas as pd
import os
os.chdir('../')
from util import get_data, plot_data
os.chdir('./Project_6')

def compute_portvals(orders_file, start_val = 1000000, commission=0, impact=0.00):
    """Accepts a df_trades df as orders_file and returns port val"""
    dates = orders_file.index
#    cumulative_holdings = orders_file.cumsum(axis=0)
    # prices data is the Adj close price per trading day
    symbols = list(orders_file.columns)
    df_prices = get_data(symbols, pd.date_range(dates[0],dates[-1]))
    if symbols.count('SPY') == 0: # SPY is kept to maintain trading days, removed if not part of portfolio, get_data adds it automatically
        df_prices = df_prices.drop('SPY', axis=1)
    
    df_prices['cash'] = 1
    
    df_holdings = orders_file.copy().cumsum()
#    df_trades['cash'] = df_trades[symbols] * df_prices[symbols] * -1
    df_holdings['cash'] = df_holdings[symbols] * df_prices[symbols] * -1
    print(df_holdings)
    
#    df_holdings = df_trades.copy()
    
#    stock_value = cumulative_holdings * prices_data
#    # df_prices columns: Date, *symbols, cash
#    # df_prices is simply price data as a dataframe
#    df_prices = pd.DataFrame(prices_data)
#    df_prices['cash'] = 1
#    
#    # df_tades is stocks traded and their respective cash value
#    df_trades = orders_file.copy()
#    df_trades['cash'] = 0
#    df_holdings = df_trades.copy()
#    df_trades['cash'] = (df_trades.iloc[:, :-1] * df_prices).sum(axis=1) * -1
#    print(df_trades)
#    
#    df_holdings.ix[0,'cash'] = start_val + df_trades.ix[0,'cash']
#    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
#    
#    df_holdings.ix[0,'cash'] = start_val + df_trades.ix[0,'cash']
#    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
#    
#    for i in range(1, df_holdings.shape[0]):
#        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]
#    print(df_holdings)
    # df_holdings represents how much money was traded
#    df_holdings = df_prices.copy()
#    df_holdings[:] = 0
#   
#    df_holdings.ix[:,:-1] = df_prices * orders_file * -1 # negative since sells are negative and increase cash
#
#    # initial value on day 0
#    df_holdings.iloc[0, -1] = start_val + df_holdings.ix[0:1, 0:-1].sum(axis=1)[0]
#    for sym in orders_file.columns:
#        if orders_file.ix[0,sym] != 0:
#            df_holdings.iloc[0,-1] -= commission + abs(orders_file.ix[0, sym] * df_prices.ix[0, sym] * impact)
#     
#    
#    for i in range(1, df_holdings.shape[0]):
#        df_holdings.iloc[i, -1] = df_holdings.iloc[i-1,-1] + df_holdings.iloc[i, 0]
#        for sym in orders_file.columns:
#            if orders_file.ix[i,sym] != 0:
#                df_holdings.iloc[i, -1] -= commission + abs(orders_file.ix[i, sym] * df_prices.ix[i, sym] * impact)
#                
#    df_portval = df_trades * df_prices 
#    df_portval.iloc[0,-1] = start_val + df_portval.iloc[0:1, 0:-1].sum(axis=1)[0]
#    for i in range(1, df_portval.shape[0]):
#        df_portval.iloc[i,-1] = df_portval.iloc[i-1,-1] + df_portval.iloc[i:i+1, 0:-1].sum(axis=1)[0]
#        
#    return df_portval['cash']
#    df_portvalue = pd.DataFrame(data={'stock_value' : stock_value.sum(axis=1), 'cash' : df_holdings['cash']}, index=dates)
#    df_portvalue['total'] = df_portvalue.sum(axis=1)
#    return df_portvalue
    
def author():
    return('mtong31')
    
def test_code():
    import numpy as np
    import datetime as dt
    from BestPossibleStrategy import bps
    
    compute_portvals(bps())
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    
#    sd = dt.datetime(2011,1,3)
#    ed = dt.datetime(2011,3,30)
#    dates = pd.date_range(sd,ed)
#    
#    df = pd.DataFrame(index=dates)
#
#    np.random.seed=42
#    fake_data = np.random.randint(-1, 2, size=df.shape[0])
#    np.random.seed=41
#    fake_data2 = np.random.randint(-1, 2, size=df.shape[0])
#    df['AAPL'] = fake_data*1000
##    df['GE'] = fake_data2*1000
#    port_value=compute_portvals(df)
#    
#    print(df)
    
#    return(df2,df)
#
#    d_returns       = port_value.copy() 
#    d_returns       = (port_value[1:]/port_value.shift(1) - 1)
#    d_returns.ix[0] = 0
#    d_returns       = d_returns[1:]
#    
#    #Below are desired output values
#    
#    #Cumulative return (final - initial) - 1
#    cr   = port_value[-1] / port_value[0] - 1
#    #Average daily return
#    adr  = d_returns.mean()
#    #Standard deviation of daily return
#    sddr = d_returns.std()
#    #Sharpe ratio ((Mean - Risk free rate)/Std_dev)
#    daily_rfr     = (1.0)**(1/252) - 1 #Should this be sampling freq instead of 252? 
#    sr            = (d_returns - daily_rfr).mean() / sddr
#    sr_annualized = sr * (252**0.5)
#
#    print(df)
#    # Compare portfolio against $SPX
#    print("Date Range: {} to {}".format(port_value.index[0], port_value.index[-1],end='\n'))
#    print("Sharpe Ratio of Fund: {}".format(sr_annualized))
##    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY,end='\n'))
#    print("Cumulative Return of Fund: {}".format(cr))
##    print("Cumulative Return of SPY : {}".format(cum_ret_SPY,end='\n'))
#    print("Standard Deviation of Fund: {}".format(sddr))
##    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY,end='\n'))
#    print("Average Daily Return of Fund: {}".format(adr))
##    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY,end='\n'))
#    print("\nFinal Portfolio Value: {}\n".format(port_value[-1]))
#    return(port_value)
    
if __name__ == "__main__":
   s = test_code()
#   print(s)