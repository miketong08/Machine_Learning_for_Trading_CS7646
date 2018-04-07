"""Randoom Tree Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""
import pandas as pd
import numpy as np
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

if __name__ == '__main__':        
    import os
    os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/data/decision_tree_data")
    test = pd.read_csv('Istanbul.csv', index_col='date')
    test = test.rename(columns={x:y for x,y in zip(test.columns, range(len(test.columns)))})
    test = np.array(test)
    x = test[:,:-1]
    y = test[:,-1]
    learner = rt.RTLearner(1)
    t = learner.addEvidence(x,y)
    a = learner.query(x)
    print(a)
    
#    test = np.array([
#            [0.61, 0.63, 8.4, 3],
#            [0.885, 0.33, 9.1, 4],
#            [0.56, 0.5, 9.4, 6],
    
#            [0.735, 0.57, 9.8, 5],
#            [0.32, 0.78, 10, 6],
#            [0.26, 0.63, 11.8, 8],
#            [0.5, 0.68, 10.5, 7],
#            [0.725, 0.39, 10.9, 5],
#        ])
#    
#    learner = RTLearner(1)
#    x = test[:,0:-1]
#    y = test[:, -1]
#    test = learner.addEvidence(x, y)
#    print(learner.tree)
#    a=learner.query(x)
#    print(a)