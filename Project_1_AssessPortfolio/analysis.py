"""Analyze a portfolio.
Python 3.6
CS7646 Project 1
Mike Tong (mtong31)
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
os.chdir("/home/mike/OMCS/CS7646-ML For Trading")

from util import get_data, plot_data


def assess_portfolio(
    sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):
    
    """
    sd = Start date (datetime format)
    ed = End date (datetime format)
    syms = List of symbols
    alllocs = allocation of syms, sums to 1.0
    sv = starting value (dollars)
    rfr = Risk free return
    sf = sampling frequency 
    """
    
    # read in adjusted closing prices for given symbols, date range
    # adding SPY to allocation for calulations and trading days
    dates = pd.date_range(sd.date(), ed.date())
    df_all = get_data(syms, dates)  # automatically adds SPY
    df = df_all[syms] 
    # get daily portfolio value    
    df_nrm          = df / df.ix[0,:] 
    allocated       = df_nrm * allocs
    position_values = allocated * sv
    port_value      = position_values.sum(axis = 1)
    
    # daily returns (y_{t} = x_{t}/x_{t-1} - 1
    d_returns       = port_value.copy() 
    d_returns       = (port_value/port_value.shift(1) - 1)[1:]
    
    # Below are desired output values
    
    # cumulative return (final - initial) - 1
    cr   = port_value[-1] / port_value[0] - 1
    # average daily return
    adr  = d_returns.mean()
    # standard deviation of daily return
    sddr = d_returns.std()
    # sharpe ratio ((Mean - Risk free rate)/Std_dev)
    daily_rfr     = (1.0 - rfr)**(1/252) - 1 #Should this be sampling freq instead of 252? 
    sr            = (d_returns - daily_rfr).mean() / sddr
    sr_annualized = sr * (sf**0.5)
    
    # compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_nrm_SPY = df_all['SPY'] / df_all['SPY'].ix[0,:]
        
        port_value_norm = port_value / port_value.ix[0,:]
        port_vs_SPY = df_nrm_SPY.copy()
        port_vs_SPY = port_vs_SPY.to_frame().join(port_value_norm.to_frame('Portfolio'))
    
        ax_portfolio = port_vs_SPY.plot(title = 'Daily Returns against SPY', grid = True, legend = 'reverse')
        ax_portfolio.set_xlabel('Date')
        ax_portfolio.set_ylabel('Normalized Price')
        plt.show()
    
    # end value
    ev = port_value[-1]
    
    return cr, adr, sddr, sr_annualized, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2011,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.1,0.2,0.3,0.4]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        rfr = risk_free_rate, \
        sf = sample_freq, \
        gen_plot = True)

    # Print statistics
    print("Start Date:          ", start_date)
    print("End Date:            ", end_date)
    print("Symbols:             ", symbols)
    print("Allocations:         ", allocations)
    print("Sharpe Ratio:        ", sr)
    print("Volatility:          ", sddr)
    print("Average Daily Return:", adr)
    print("Cumulative Return:   ", cr)
    
if __name__ == "__main__":
    test_code()
