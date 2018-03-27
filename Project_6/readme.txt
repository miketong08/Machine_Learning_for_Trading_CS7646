To generate a trading scheme with ManualStrategy.py, the following libraries are needed:

pandas
numpy
datetime
indicators.py
util.py

The following are optional, but return plots and metrics.

matplotlib.pyplot
marketsimcode.py
BestPossibleStrategy.py

*The submitted code can be run directly to generate a plot of the manual learner vs benchmark for the insample criteria.

For generation of a trading scheme, the function ManualStrategy.testPolicy(symbol, sd, ed) is called, which accepts a symbol, start date (sd), and end date (ed)
The default arguments are JPM, 01-01-2008, 12-31-2009.

ManualStrategy.testPolicy() returns an orders dataframe representing trades for each trading day in the range, which is represented by the number of shares and positive being a purchase, negative being a sell.

marketsimcode.compute_portvals() can then be called on the orders dataframe to generate a dataframe representing the portfolio's value over time.

For the best possible strategy, BestPossibleStrategy.testPolicy(symbol, sd, ed) works identically to the ManualStrategy.testPolicy(), and also returns a single column dataframe of trades.


To display basic metrics, as produced in Project 1: Assess Learners, ManaulStrategy.print_stats() can be called with the portfolio value dataframe as the argument. This will return a printout of the sharpe ratio, cumulative returns, volatility, average daily rate of return, and the final portfolio value.

There are also two plotting functions in the ManualStrategy.py, namely plot_indicators and plot_oscillators. 
Both of these simply plots information from the respective dataframes.


