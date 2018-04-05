"""Insane Bag Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""

import numpy as np
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, bag_learner=bl.BagLearner, learner=dt.DTLearner, kwargs = {}, bags=20, boost=False, verbose=False):
        
        bag_learners = []
        for i in range(bags):
            bag_learners.append(bag_learner(learner, kwargs=kwargs, bags=bags))
            
        self.bag_learners = bag_learners
        self.boost = boost #future project
        self.verbose = verbose
        
    def addEvidence(self, Xtrain, Ytrain):
        for learn in self.bag_learners:
            learn.addEvidence(Xtrain, Ytrain)
    
    def query(self,Xtest):
        outputs = [i.query(Xtest) for i in self.bag_learners]
        return(np.average(outputs,axis=0))
            
if __name__ =="__main__":
#    import os
#    import pandas as pd
#    os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/data/decision_tree_data")
#    test = pd.read_csv('Istanbul.csv', index_col='date')
#    test = test.rename(columns={x:y for x,y in zip(test.columns, range(len(test.columns)))})
#    test = np.array(test)
#    x = test[:,:-1]
#    y = test[:,-1]
#    learner = InsaneLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
#    t = learner.addEvidence(x,y)
#    a = learner.query(x[:10])
#    print(a)
    
    test = np.array([
            [0.61, 0.63, 8.4, 3],
            [0.885, 0.33, 9.1, 4],
            [0.56, 0.5, 9.4, 6],
            [0.735, 0.57, 9.8, 5],
            [0.32, 0.78, 10, 6],
            [0.26, 0.63, 11.8, 8],
            [0.5, 0.68, 10.5, 7],
            [0.725, 0.39, 10.9, 5],
        ])
    learner = InsaneLearner(learner=dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
    x = test[:,0:-1]
    y = test[:, -1]
    test = learner.addEvidence(x, y)
    #print(learner.trees)
    a=learner.query(x)
    print(a)