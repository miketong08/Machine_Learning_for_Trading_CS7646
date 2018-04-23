"""Indicators
Python 3.6
CS7646 Project 8 - Strategy Learner
Michael Tong (mtong31)

This file provides technical indicators for use in the Strategy Learner trader.
"""

import pandas as pd
from util import get_data, plot_data

def simple_ma(df, window=5, bollinger=False, threshold=2):
    """Takes a dataframe and returns a df with simple moving average values for the given window
    Optionally adds Bollinger Bands as a tuple (mean, upper-band, lower-band)
    """
    mean = df.rolling(window,center=False).mean()
    
    if bollinger:
        std = df.rolling(window, center=False).std()
        upper = mean + threshold*std
        upper.rename(columns={mean.columns[0] : 'upper_band'}, inplace=True)
        
        lower = mean - threshold*std
        lower.rename(columns={mean.columns[0] : 'lower_band'}, inplace=True)
        
        mean.rename(columns={mean.columns[0] : 'mean'}, inplace=True)
        return((mean, upper, lower))
    
    mean.rename(columns={mean.columns[0] : 'mean'}, inplace=True)
    return(mean)
        
def exp_ma(df, days=12):
    """Takes a df and number of days and returns an exponential moving average"""
    ema = df.ewm(com=((days-1)/2)).mean()
    ema.rename(columns={ema.columns[0] : 'exp_ma'}, inplace=True)
    return(ema)
    
def MACD(df, ema1_days=12, ema2_days=26, macd_signal_days=9):
    """Accepts a df, returns a df with MACD - MACD_singal, where MACD is (ema1-ema2) and MACD_signal is the EMA of MACD"""
    ema1 = exp_ma(df, ema1_days)
    ema2 = exp_ma(df, ema2_days)
    
    macd = ema1-ema2
    macd_signal = exp_ma(macd, macd_signal_days)
    
    macd_signal.rename(columns={macd.columns[0] : 'macd_signal'}, inplace=True)
    macd.rename(columns={macd.columns[0] : 'macd'}, inplace=True)
    return(macd,macd_signal)
    
def stoc_osc(df, k_window=14, d_window=3):
    """Returns the current market rate of the main and average lines respectively"""
    high = df.rolling(window=k_window, center=False).max()
    low = df.rolling(window=k_window, center=False).min()
    K = 100*(df-low)/(high-low)
    D = K.rolling(window=d_window, center=False).mean()
    
    K.rename(columns={K.columns[0] : 'K'}, inplace=True)
    D.rename(columns={D.columns[0] : 'D'}, inplace=True)
    return(K,D)
    
def CCI(df, window=20):
    moving_average = df.rolling(window, center=False).mean()
    moving_std = df.rolling(window, center=False).std()
    cci = (df-moving_average)/(0.015 * moving_std)
    return(cci.rename(columns={cci.columns[0] :'CCI'}))

def on_balance_volume(df):
    """Accepts a adjusted closed df, returns a df of obv's"""
    volume = get_data([df.columns[0]], pd.date_range(df.index[0], df.index[-1]), colname='Volume')    
    
    if df.columns[0] != 'SPY':
        volume.drop('SPY', axis=1, inplace=True)
    volume.rename(columns={volume.columns[0] : 'Vol'}, inplace=True)
    obv = volume.copy()

    price_diff = df.diff(1)
    price_diff.reset_index(drop=True, inplace=True)

    for index, series in price_diff.iloc[1:, 0].iteritems():
        if series < 0:
            obv.iloc[index, 0] = obv.iloc[index-1, 0] - obv.iloc[index, 0]
        elif series > 0:
            obv.iloc[index, 0] += obv.iloc[index-1, 0]
        else:
            obv.iloc[index, 0] = obv.iloc[index-1, 0]
    
    return(obv)
    
def author():
    return('mtong31')



