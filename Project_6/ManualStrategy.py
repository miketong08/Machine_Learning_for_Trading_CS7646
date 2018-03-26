"""
CS7646 ML For Trading
Project 6: Manual Strategy
Manual Strategy Function
Michael Tong (mtong31)
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import warnings

from indicators import simple_ma, exp_ma, MACD, stoc_osc
from marketsim import compute_portvals
from BestPossibleStrategy import testPolicy as bps
from util import get_data, plot_data

def testPolicy(df):
    """Returns a trade dataframe based on BB, MACD, and Stoch Osc"""
    # BB was implemented into the SMA, the threshold value had been iteratively optimized
    sma, upper, lower = simple_ma(df, window=19, bollinger=True, threshold=1.45102)
    
    exp12 = exp_ma(df, days=12)
    exp12 = exp12.rename(columns={exp12.columns[0] : 'ema12'})
    
    exp24 = exp_ma(df, days=24)
    exp24 = exp24.rename(columns={exp24.columns[0] : 'ema24'})
    
    exp_diff = (exp24['ema24'] - exp12['ema12']).rename('ema_diff')
    exp_diff = exp_diff.to_frame()
#    ema_diff = (exp_diff.shift(1)/exp_diff)/abs(exp_diff.shift(1)/exp_diff)

    macd, macd_s = MACD(df, ema1_days=12, ema2_days=24, macd_signal_days=9)
    k, d = stoc_osc(df, k_window=12, d_window=3)
    
    md = (macd['macd'] - macd_s['macd_signal']).rename('m-d')
    md = md.to_frame()
    md_diff = (md.shift(1)/md)/abs(md.shift(1)/md) # checks if MACD crosses its signal (-1 if it does)
    
    kd = (k['K'] - d['D']).rename('k-d')
    kd = kd.to_frame()
    kd_diff = (kd.shift(1)/kd)/abs(kd.shift(1)/kd) # checks if k-d crosses zero (-1 if it does)

    # df2 is a dashboard of indicator values
    df2 = pd.concat([df, sma, exp12, exp24, upper, lower, macd, macd_s, md, k, d, kd], axis=1)
    
    # generating a long/short df marked with 1's or 0's, 1's meaning perform the action
    df_buy = pd.DataFrame(index=df2.index)
    df_sell = df_buy.copy()
    
    df_buy['BB'] = np.where(df2.ix[:,0] < df2.ix[:,'lower_band'], 1, 0)
    df_buy['Stoch_D'] = np.where(df2.ix[:,'D'] < 30, 1, 0)
    df_buy['KD'] = np.where(kd_diff.ix[:,0] == -1, 1, 0)
    df_buy['MACD'] = np.where(md_diff.ix[:,0] == -1, 1, 0)
#    df_buy['ema_diff'] = np.where(ema_diff.ix[:,0] == -1, 1, 0)
    
    df_sell['BB'] = np.where((df2.ix[:,0] > df2.ix[:,'upper_band']), 1, 0)
    df_sell['Stoch_D'] = np.where(df2.ix[:,'D'] > 70, 1, 0)
    df_sell['KD'] = df_buy['KD']
    df_sell['MACD'] = df_buy['MACD']
#    df_sell['ema_diff'] = df_buy['ema_diff']
    
    df_trades = pd.DataFrame(index=df.index)
    df_trades[df.columns[0]] = 0
    holding = 0
    
    # Trading Scheme is to long/short primarily off BB crossings. The second criteria is off the strength of the momentum indicators
    for i in df_trades.index:
        if holding == 0:
            if df_buy.ix[i, 'BB'] == 1 and df_buy.ix[i, 'Stoch_D'] == 1 and (df_buy.ix[i, 'KD'] == 1 or df_buy.ix[i, 'MACD'] ==1):
                df_trades.ix[i,0] = 1000
                holding = 1000
            elif df_sell.ix[i, 'BB'] == 1 and df_sell.ix[i, 'Stoch_D'] == 1 and (df_sell.ix[i, 'KD'] == 1 or df_sell.ix[i, 'MACD'] ==1):
                df_trades.ix[i,0] = -1000
                holding = -1000
        elif holding == 1000:
            if df_sell.ix[i, 'BB'] == 1 and df_sell.ix[i, 'Stoch_D'] == 1 and (df_sell.ix[i, 'KD'] == 1 or df_sell.ix[i, 'MACD'] ==1):
                df_trades.ix[i,0] = -2000
                holding = -1000
        
        elif holding == -1000:
            if df_buy.ix[i, 'BB'] == 1 and df_buy.ix[i, 'Stoch_D'] == 1 and (df_buy.ix[i, 'KD'] == 1 or df_buy.ix[i, 'MACD'] ==1):
                df_trades.ix[i,0] = 2000
                holding = 1000
                
    return(df_trades)
    
def plot_indicators(df, *args):
    """Plots indicators of a given df, returns the df with indicators"""
    fig = plt.figure(figsize=(10,5), dpi=120)
    plt.plot(df, color='b', label='Share Price')
    plt.rcParams.update({'font.size': 16})
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Share Price', fontsize=18)
    plt.xticks(rotation=45)
    fig.suptitle('SMA of Share Price')
    
    if len(args)>0:
        for i in args:
            plt.plot(i, label=i.name, color='y')
            
    fig.legend(loc=4, bbox_to_anchor=(0.85,0.25))            
    plt.show()
    
def plot_oscillator(df1, df2, title):
    """Simple function that accepts two df's and a plot title, returns a plot of the oscillators"""
    fig = plt.figure(figsize=(10,5), dpi=120)
    plt.plot(df1, color='y', label=df1.name)
    plt.plot(df2, color='b', label=df2.name)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Arbitrary Price Percentage', fontsize=18)
    plt.xticks(rotation=45)
    fig.suptitle(title, fontsize=24)
    
    fig.legend(loc=4, bbox_to_anchor=(0.85,0.25))
    plt.show()
    
def optimize_BB(df, w, th):
    """Crude iterative optimization method for BB window and threshold"""
    trades = testPolicy(df,w,th)
    port_val = compute_portvals(trades)[-1]
    best = port_val
    new_th = None
    new_w = None
    for i in np.linspace(0.1, th, 50):
        for j in range(17, 22):
            if compute_portvals(testPolicy(df,j, i))[-1] > best:
                best = compute_portvals(testPolicy(df,j, i))[-1]
                new_th = i
                new_w = j
                print(new_w, new_th, end='\n\n')
                
def optimize_macd(df,ema1,ema2,macds):
    """Crude iterative optimization method for MACD EMA days and signal days"""
    trades = testPolicy(df,ema1,ema2,macds)
    port_val = compute_portvals(trades)[-1]
    best = port_val
    for e1 in range(ema1-3, ema1+3):
        for e2 in range(ema2-3, ema2+3):
            for s in range(macds-3, macds+3):
                if compute_portvals(testPolicy(df,e1,e2,s))[-1] > best:
                    best = compute_portvals(testPolicy(df,e1,e2,s))[-1]
                    print(e1, e2, s)
    
def optimize_stoch(df, k, d):
    """Crude iterative optimization method for stochastic window days"""
    trades = testPolicy(df,k,d)
    port_val = compute_portvals(trades)[-1]
    best = port_val
    for k1 in range(k-6, k+6):
        for d1 in range(d-2, d+5):
            if compute_portvals(testPolicy(df,k1,d1))[-1] > best:
                best = compute_portvals(testPolicy(df,k1,d1))[-1]
                print(k1,d1)

def print_stats(df):
    """Accepts a df of portfolio values and returns basic metrics of the portfolio, sharpe ratio, cumulative returns, 
    volatility, average daily return, and final portfolio value"""
    port_value = df
    d_returns       = port_value.copy() 
    d_returns       = (port_value[1:]/port_value.shift(1) - 1)
    d_returns.iloc[0] = 0
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
    print("\nDate Range: {} to {}".format(port_value.index[0], port_value.index[-1],end='\n'))
    print("Sharpe Ratio of Fund: {}".format(sr_annualized))
    print("Cumulative Return of Fund: {}".format(cr))
    print("Standard Deviation of Fund: {}".format(sddr))
    print("Average Daily Return of Fund: {}".format(adr))
    print("\nFinal Portfolio Value: {}\n".format(port_value[-1]))
    
def author():
    return('mtong31')
    
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    df = get_data(['JPM'], pd.date_range(sd,ed))
    df_spy = df['SPY']
    df.drop('SPY',axis=1,inplace=True)
    df_normed = df/df.ix[0]
    
    b = bps('JPM',sd,ed)
    best = compute_portvals(b)
    
    manual = testPolicy(df)
    value = compute_portvals(manual)
    
    df_spy_hold = pd.DataFrame(index=df_spy.index)
    df_spy_hold['SPY'] = 0
    df_spy_hold.iloc[0,0] = int((2000*df.iloc[0,0])/df_spy.iloc[0])
    
    df_spy_val = compute_portvals(df_spy_hold)
    
    
#    value = value/value[0]
#    best = best/best[0]
    
##      Below is for plot generation    
#    bv = bps('JPM',sd,ed)    
#
#    fig = plt.figure(figsize=(10,6), dpi=120)
#    plt.rcParams.update({'font.size': 16})
#    plt.xlabel('Date', fontsize=18)
#    plt.xticks(rotation=45)
    
##      Below is for performance
#    plt.plot(value, color='k')
#    fig.suptitle('Portfolio Performance')    
#    plt.ylabel('Proportion Gain')
#    plt.legend(['Manual Strategy'], loc=4, prop={'size' : 15})    

#    plt.plot(b, color='b')    
#    fig.suptitle('Best Possible Performance')
#    plt.ylabel('Proportion Gain')
#    plt.legend(['Best Possible Strategy'], loc=4, prop={'size':15})
##
#    plt.plot(best/best[0], color='b', label='Best Strategy')    
#    plt.plot(value/value[0], color='k', label='Manual Strategy')
#    plt.plot(df_spy_val/df_spy_val[0], color= 'y', label='SPY')
#    fig.suptitle('Best Strategy vs Manual Strategy vs SPY')
#    plt.ylabel('Proportional Gain')
#    plt.legend(loc=7, prop={'size':15})

#    plt.plot(df,color='k')
#    fig.suptitle('In Sample Long/Short Positions', fontsize=20)
#    plt.ylabel('Share Price', fontsize=16)
#    plt.legend(['Manual Strategy'], loc=4, prop={'size' : 15})

##      Below is to show vert lines
#    for i in manual[manual[manual.columns[0]] != 0].index:
#        if manual.ix[i,0] > 0:
#            plt.axvline(x=i, color='g')
#        elif manual.ix[i,0] < 0:
#            plt.axvline(x=i, color='r')

    plt.show()
    
