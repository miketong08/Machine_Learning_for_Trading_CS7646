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
os.chdir('./Project_5')

def compute_portvals(orders_file = "./data/", start_val = 1000000, commission=9.95, impact=0.005):
    """Returns a single column df with porfolio values from the beginning to the end of the order file"""
    # orders columns: Date, symbol, order, shares
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True) 
    orders = orders.sort_index()

    symbols = list(set(orders['Symbol'])) 
    dates = list(set(orders.index)) # dates where things actually happen
    dates.sort()
    
    # prices data is the Adj close price per trading day
    prices_data = get_data(symbols, pd.date_range(dates[0],dates[-1]))
    if symbols.count('SPY') == 0: # SPY is kept to maintain trading days, removed if not part of portfolio, get_data adds it automatically
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
    
    date_commission = dict([[d, 0] for d in dates]) # keeps track of comission and market impact per date as key
    for index, col in orders.iterrows():
        if col['Order'] == 'SELL':    
            df_trades.loc[index, col['Symbol']] += col['Shares'] * -1 
        else:
            df_trades.loc[index, col['Symbol']] += col['Shares'] 
            
        # used orders as opposed to df_trades because same day trades may result in zero shares for the particular day
        date_commission[index] -= commission + (col['Shares'] * df_prices.loc[index, col['Symbol']] * impact) 
        
    for index in dates:
        df_trades.loc[index, 'cash'] += -1*(df_trades.ix[index, :-1].multiply(df_prices.ix[index, :-1]).sum())\
                                         + date_commission[index] 
                                         
    df_holdings.ix[0,'cash'] = start_val + df_trades.ix[0,'cash']
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]
    
    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]
    print('DF_HOLDINGS')
    print(df_holdings)
    print('DF TRADES')
    print(df_trades)
    # df_value is the value of each holding and total as a dollar amount
    df_value = df_holdings.multiply(df_prices)
    df_portval = df_value.sum(axis=1)
    return df_portval
    
def author():
    return('mtong31')
    
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "../../data/orders/orders-10.csv" #1026658.3265
#    of = "./orders/orders-11.csv"
#    of = "./orders/orders-12.csv" #1705686.6665
    sv = 1000000

    # Process orders
    if of == "../../data/orders/orders-12.csv":
        port_value = compute_portvals(orders_file = of, start_val = sv, commission=0)
    else:
        port_value = compute_portvals(orders_file = of, start_val = sv, commission=9.95)
        
    if isinstance(port_value, pd.DataFrame):
        port_value = port_value[port_value.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

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
    
    return(compute_portvals(orders_file = of, start_val = sv))
if __name__ == "__main__":
   s = test_code()