"""MC2-P1: Market simulator.
orders.csv: https://youtu.be/1ysZptg2Ypk?t=3m48s
df_prices: https://youtu.be/1ysZptg2Ypk?t=6m30s
df_trades: https://youtu.be/1ysZptg2Ypk?t=9m9s
df_holdings: https://youtu.be/1ysZptg2Ypk?t=15m47s
"""

import pandas as pd
import os
os.chdir('../')
from util import get_data, plot_data
os.chdir('./Project_6')

def compute_portvals(orders_file, start_val = 100000, commission=9.95, impact=0.005):
    """Returns a single column df with porfolio values from the beginning to the end of the order file"""
    
    dates = orders_file.index
    symbol = orders_file.columns[0]
    # prices data is the Adj close price per trading day
    prices_data = get_data([symbol], pd.date_range(dates[0],dates[-1]))
    if symbol != 'SPY': # SPY is kept to maintain trading days, removed if not part of portfolio, get_data adds it automatically
        prices_data = prices_data.drop('SPY', axis=1)
        
    # df_prices columns: Date, *symbols, cash
    # df_prices is simply price data as a dataframe
    df_prices = pd.DataFrame(prices_data)
    df_prices['cash'] = 1
    
    # df_trades represents number of shares held and cash avalable only on order dates
    df_trades = df_prices.copy()
    df_trades[:] = 0
    
    # df_holdings represents df_trades, but on every date between sd and ed
    df_holdings = df_trades.copy()    
    
    for i in orders_file.index:
        if orders_file.ix[i,symbol] != 0:
            df_trades.ix[i, symbol] = orders_file.ix[i, symbol]
            df_trades.ix[i, 'cash'] = -1 * orders_file.ix[i, symbol] * df_prices.ix[i, symbol]\
                                      - abs(commission + orders_file.ix[i, symbol] * df_prices.ix[i, symbol] * impact) 

    df_holdings.ix[0,'cash'] = start_val + df_trades.ix[0,'cash']
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
    
    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]
#    # df_value is the value of each holding and total as a dollar amount
    df_value = df_holdings.multiply(df_prices)
    df_portval = df_value.sum(axis=1)
    return(df_portval)
    
def author():
    return('mtong31')
    
def test_code():
    from BestPossibleStrategy import bps
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    port_value = compute_portvals(bps())

    d_returns       = port_value.copy() 
    d_returns       = (port_value[1:]/port_value.shift(1) - 1)
    d_returns.ix[0] = 0
    d_returns       = d_returns[1:]
    
    #Below are desired output values
    
    #Cumulative return (final - initial) - 1
    cr   = port_value[-1] / port_value[0] - 1
    #Average daily return
    adr  = d_returns.mean()
    #Standard deviation of daily return
    sddr = d_returns.std()
    #Sharpe ratio ((Mean - Risk free rate)/Std_dev)
    daily_rfr     = (1.0)**(1/252) - 1 #Should this be sampling freq instead of 252? 
    sr            = (d_returns - daily_rfr).mean() / sddr
    sr_annualized = sr * (252**0.5)


    # Compare portfolio against $SPX
    print("Date Range: {} to {}".format(port_value.index[0], port_value.index[-1],end='\n'))
    print("Sharpe Ratio of Fund: {}".format(sr_annualized))
#    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY,end='\n'))
    print("Cumulative Return of Fund: {}".format(cr))
#    print("Cumulative Return of SPY : {}".format(cum_ret_SPY,end='\n'))
    print("Standard Deviation of Fund: {}".format(sddr))
#    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY,end='\n'))
    print("Average Daily Return of Fund: {}".format(adr))
#    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY,end='\n'))
    print("\nFinal Portfolio Value: {}\n".format(port_value[-1]))
    
    
if __name__ == "__main__":
   s = test_code()