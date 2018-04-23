"""Strategy Learner
Python 3.6
CS7646 Project 8 - Strategy Learner
Mike Tong (mtong31)

This program returns a trades dataframe based on a random forest classifier
"""

import pandas as pd
import datetime as dt
import numpy as np
import RTLearner as rt
import BagLearner as bl
import indicators as ind
import util 

# below is for local file operation
import os
os.chdir('/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/Project_8_StrategyLearner')
#####

class StrategyLearner(object):
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose # aids in debugging
        self.impact = impact
        self.learner = None
        self.indicator_window = 20 # adjusts window size for rolling indicators
    
    def author(self):
        return('mtong31')
        
    # if specific conditions for the learner want to be modified, use the function below prior to training
    def get_tree(self, learner=rt.RTLearner, leaf_size=6, bags=18): # 6 , 20
        """Initalizes the learner parameters, does not return anything, but primes the learner for training"""
        self.learner = bl.BagLearner(learner, kwargs={'leaf_size' : leaf_size}, bags=bags, boost=False)
        
    def signage(self, value):
        """Accepts a value, returns the normalized value as -1 or 1"""
        return(value/abs(value))
    
    def df_prices(self, symbol, sd, ed):
        """Returns the prices dataframe from util.get_data"""
        df_prices = util.get_data([symbol], pd.date_range(sd, ed))
        # SPY is kept to maintain trading days, removed if not part of portfolio, get_data adds it automatically
        if symbol != 'SPY': 
            df_prices = df_prices.drop(['SPY'], axis=1)
        return(df_prices)
        
    def feature_balance(self, trees):
        """Function to check the overall balance of the formed trees by counting the number of features used for split values"""
        num_columns = len(self.xTrain.columns)
        feature_count = dict(enumerate([0]*num_columns))
        
        for tree in trees:
            for branch in tree:
                if branch[0] > -1:
                    feature_count[branch[0]] += 1
                    
        for i in range(num_columns):
            feature_count[self.xTrain.columns[i]] = feature_count.pop(i)
        return(feature_count)
    
    def df_yTrain(self, symbol, sd, ed):
        """Creates buy/hold/sell signals based on price history. Impact affects trading thresholds. Returns a trades dataframe"""
        df_prices = self.df_prices(symbol, sd, ed)
        df_trades = pd.DataFrame(data=0, index=df_prices.index, columns={symbol})

        df_prices['diff'] = df_prices.diff(-1)
        df_prices.iloc[0, -1] = 0
        df_prices.fillna(method='ffill',inplace=True) # if successive prices dont change, fills with the price prior to it.
        
        prev_pos = df_prices.iloc[0,-1]
        df_trades[symbol] = prev_pos
        
        for i,j in df_prices[1:].iterrows():
            if j['diff'] == prev_pos:
                df_trades.loc[i] = 0
            elif j['diff'] < 0 and j['diff'] > -self.impact*df_prices.loc[i,symbol]:
                df_trades.loc[i] = 0
            elif j['diff'] > 0 and j['diff'] < self.impact*df_prices.loc[i,symbol]:
                df_trades.loc[i] = 0
            else:
                prev_pos = j['diff']
                df_trades.loc[i] = prev_pos * -1
        
        df_trades = self.signage(df_trades.iloc[:,0])
        df_trades.fillna(0, inplace=True)
        df_trades[-1] = 0 # last day cannot see into the future
        
        if self.verbose:
            print(pd.concat([df_prices, df_trades],axis=1))
            
        return(df_trades)
        
    # indicator_values is used to generate the input data for the model
    def indicator_values(self, df_prices):
        """Function is used to generate X inputs based on indicator performance
        Accepts a prices df from util.get_data, returns a df of indicator performance"""
#        window = self.indicator_window
        
        # simple moving average and bollinger bands
        sma, upper, lower = ind.simple_ma(df_prices, window=19, bollinger=True, threshold=1.45102)
        
        # BB percentage
        perc_upper = df_prices.iloc[:,0] / upper.iloc[:,0] 
        perc_upper = pd.DataFrame(perc_upper.rename('upper_BB%'))
        perc_lower = lower.iloc[:,0] / df_prices.iloc[:,0]
        perc_lower = pd.DataFrame(perc_lower.rename('lower_BB%'))
                
        # moving average convergence divergence
        macd, macd_s = ind.MACD(df_prices, ema1_days=12, ema2_days=24)

        # below checks percentage crossover
        md = macd_s['macd_signal'] / macd['macd']
        md.rename('macd_sigal/macd')
        
        # stoichastic oscillator
        K, D = ind.stoc_osc(df_prices, k_window=12)
        # below is to check for percentage of crossover for the stoich osc
        k = D.iloc[:,0] / K.iloc[:,0]
        k.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        k.rename('D/K', inplace=True)
                
        df_ind = pd.concat([perc_upper, perc_lower, md, D, k], axis=1)
        self.window = df_ind.isnull().any(1).nonzero()[0].max() + 1 # highest index is mainteined to prevent training during this period
        
        if self.verbose:
            self.ind = df_ind
        
        return(df_ind.iloc[self.window:,:])

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 
        "Accepts a symbol, start date (sd), end date(ed) and start value (sv), creates an optimized decision tree"""
        df_prices = self.df_prices(symbol, sd, ed)

        xTrain = self.indicator_values(df_prices)
        self.yTrain = self.df_yTrain(symbol, sd, ed)[self.window:]
        
        if len(self.yTrain.nonzero()[0]) != 0:
            if self.learner == None:
                self.get_tree()
            self.learner.addEvidence(xTrain.values, self.yTrain.values.astype('int8'))
        else:
            if self.verbose: print('Impact is probably too high')

        self.xTrain = xTrain
        
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        """Creates a trading dataframe based on queried indicator values for the out of sample data using a random forest"""
        df_prices = self.df_prices(symbol, sd, ed)
        
        if len(self.yTrain.nonzero()[0]) == 0:
            return(pd.DataFrame(data=0, index=df_prices.index,columns=[symbol]))
            
        xTest = self.indicator_values(df_prices)
        
        query = self.learner.query(xTest.values)
        
        query[query <= -0.5] = -2000
        query[query >= 0.5] = 2000
        query[abs(query) != 2000] = 0       
        
        query = np.insert(query, 0, np.zeros(self.window)) # replaces rolling window values with zero to prevent querying
        
        if len(query.nonzero()[0]) == 0:
            if self.verbose: print('Impact is definitely too high')
            return(pd.DataFrame(data=0, index=df_prices.index,columns=[symbol]))

        first_trade = query.nonzero()[0][0]
        if first_trade > 0:
            query[first_trade] = 1000
        else:
            query[first_trade] = -1000

        current_pos = self.signage(query[first_trade])
        
        for pos in range(first_trade+1, len(query)):
            if query[pos] == 0:
                continue
            elif self.signage(query[pos]) == current_pos:
                query[pos] = 0
            else:
                query[pos] = self.signage(query[pos]) * 2000
                current_pos = self.signage(query[pos])

        df_trades = pd.DataFrame(index=df_prices.index)
        df_trades[symbol] = query
        
        if self.verbose:
            self.query=query
            self.xTest = xTest

        return(df_trades)
    