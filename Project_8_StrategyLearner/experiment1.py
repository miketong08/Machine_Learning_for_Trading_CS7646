"""experiment1
Python 3.6
CS7646 Project 8 - Strategy Learner
Mike Tong (mtong31)
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode
import StrategyLearner as sl
import ManualStrategy as ms

"""This code compares the performance of the strategy learner to the project 6 manual strategy, as well as the 
respective benchmark (buy and hold 1000 shares).

The code will print a plot of the normalized gain, as well as a dataframe with the respective values"""
def author():
    return('mtong31')
    
if __name__ == "__main__":
    print('Project 8 Experiment1: mtong31')
    sym = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sd2 = dt.datetime(2010,1,1)
    ed2 = dt.datetime(2011,12,31)
    sv = 100000
    learner = sl.StrategyLearner(verbose=False, impact=0.00)
    learner.addEvidence(sym, sd, ed, 100000)
    
    strategy = learner.testPolicy(sym, sd, ed)
    manual = ms.testPolicy(sym, sd, ed)
    benchmark = pd.DataFrame(index=strategy.index)
    benchmark[sym] = 0
    benchmark.iloc[0,0] = 1000    
    
    values = marketsimcode.compute_portvals(strategy)
    values_bench = marketsimcode.compute_portvals(benchmark)
    values_manual = marketsimcode.compute_portvals(manual)
    
    # below normalizes gains to the sv
    values /= sv
    values_bench /= sv
    values_manual /= sv
    
    fig = plt.figure(figsize=(10,5), dpi=80)
    plt.plot(values, color='b', label='Strategy')
    plt.plot(values_bench, color='r', linestyle=':', linewidth=2, label='Benchmark')
    plt.plot(values_manual, color='y', label='Manual')
    plt.xlabel('Dates', fontsize=14)
    plt.ylabel('Portfolio value', fontsize=14)
    
    fig.suptitle('In Sample Comparison: Benchmark vs Manual vs Strategy', fontsize=18)
    fig.legend(loc=3, bbox_to_anchor=(0.08, 0.7))
    plt.show()
    
    columns = len(learner.xTrain.columns)
    result_df = pd.concat([values, values_bench, values_manual],axis=1)
    result_df.rename(columns={0 : 'Strategy_Learner', 1 : 'Benchmark', 2 : 'Manual_Strategy'}, inplace=True)
    print('Performance comparison: ')
    print(result_df)