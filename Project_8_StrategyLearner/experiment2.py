"""experiment2
Python 3.6
CS7646 Project 8 - Strategy Learner
Mike Tong (mtong31)
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode
import StrategyLearner as sl

"""This code compares the performance of the strategy learner under various impact values as well as the 
respective benchmark (buy and hold 1000 shares).

The code will print a plot of the normalized gain, as well as a dataframe with the respective values"""
def author():
    return('mtong31')
    
if __name__ == "__main__":
    print('Project 8 Experiment2: mtong31')
    sym = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sd2 = dt.datetime(2010,1,1)
    ed2 = dt.datetime(2011,12,31)
    sv = 100000
    
    learner  = sl.StrategyLearner(verbose=False, impact=0.00)
    learner2 = sl.StrategyLearner(verbose=False, impact=0.0005)
    learner3 = sl.StrategyLearner(verbose=False, impact=0.002)
    learner4 = sl.StrategyLearner(verbose=False, impact=0.005)
    learner5 = sl.StrategyLearner(verbose=False, impact=1.00)

    # Training phase
    learner.addEvidence(sym, sd, ed, 100000)
    learner2.addEvidence(sym, sd, ed, 100000)
    learner3.addEvidence(sym, sd, ed, 100000)
    learner4.addEvidence(sym, sd, ed, 100000)
    learner5.addEvidence(sym, sd, ed, 100000)
    
    # Test Phase
    strategy  = learner.testPolicy(sym, sd, ed)
    strategy2 = learner2.testPolicy(sym, sd, ed)
    strategy3 = learner3.testPolicy(sym, sd, ed)
    strategy4 = learner4.testPolicy(sym, sd, ed)
    strategy5 = learner5.testPolicy(sym, sd, ed)
    
    # Performance Metrics
    values  = marketsimcode.compute_portvals(strategy)
    values2 = marketsimcode.compute_portvals(strategy2)
    values3 = marketsimcode.compute_portvals(strategy3)
    values4 = marketsimcode.compute_portvals(strategy4)
    values5 = marketsimcode.compute_portvals(strategy5)

    # Normalize gains to the sv
    values  /= sv
    values2 /= sv
    values3 /= sv
    values4 /= sv
    values5 /= sv
    
    fig = plt.figure(figsize=(10,5), dpi=80)
    plt.plot(values,  label='impact: 0.00')
    plt.plot(values2, label='impact: 0.0005')
    plt.plot(values3, label='impact: 0.002')
    plt.plot(values4, label='impact: 0.005')
    plt.plot(values5, label='impact: 1.00')
    
    plt.xlabel('Dates', fontsize=14)
    plt.ylabel('Portfolio value', fontsize=14)
    fig.suptitle('Impact value comparison', fontsize=18)
    fig.legend(loc=3, bbox_to_anchor=(0.08, 0.6))
    plt.show()
    
    columns = len(learner.xTrain.columns)
    result_df = pd.concat([values, values2, values3, values4, values5],axis=1)
    result_df.rename(columns={0 : 'impact: 0.00',\
                              1 : 'impact: 0.0005',\
                              2 : 'impact: 0.002',\
                              3 : 'impact: 0.005',\
                              4 : 'impact: 1.00'}, inplace=True)
    
    print('Performance comparison: ')
    print(result_df)