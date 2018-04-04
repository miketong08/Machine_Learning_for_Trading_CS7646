"""Bag Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""

import numpy as np
import pandas as pd
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):
    def __init__(self, learner, kwargs = {}, bags=20, boost=False, verbose=False):
        
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
            
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        #self.boost = boost
        self.verbose = verbose
        self.trees = []

    def author(self):
        return('mtong31')
    
    def addEvidence(self, Xtrain, Ytrain):
        df = pd.DataFrame(Xtrain)
        df['output'] = Ytrain
        
        for method in self.learners:
            learning_df = pd.DataFrame([df.sample().values[0] for i in range(df.shape[0])])
            X = learning_df.iloc[:, :-1]
            Y = learning_df.iloc[:,-1]
            method.addEvidence(X,Y)
            self.trees.append(method.tree)
            
    def query_value(self, values, tree):
        """Queries a single list of values for a given tree, returns the output of the tree"""
        current_pos = 0
        while True:
            tree_pos = tree[current_pos]
            if current_pos > tree.shape[0]: 
                return('Error querying value')
            elif int(tree_pos[0]) == -1: 
                return(tree_pos[1])            
            elif values[int(tree_pos[0])] <= tree_pos[1]: 
                current_pos += 1
            else: 
                current_pos += int(tree_pos[3]) 
            
    def query_trees(self,Xtest, tree):
        """Given an input (Xtest), returns the associated query output(s), can accept arrays"""
        try: # assumes multiple test values
            return([self.query_value(i, tree) for i in Xtest])
                
        except:
            return([self.query_value(Xtest, tree)])
        
    def query(self,Xtest):
        queries = [self.query_trees(Xtest, i) for i in self.trees]
        return(np.average(queries,axis=0))
            
if __name__ =="__main__":
#    import os
#    os.chdir("/home/mike/OMCS/CS7646-ML For Trading/CS7646_Assignments/data/decision_tree_data")
#    test = pd.read_csv('Istanbul.csv', index_col='date')
#    test = test.rename(columns={x:y for x,y in zip(test.columns, range(len(test.columns)))})
#    test = np.array(test)
#    x = test[:,:-1]
#    y = test[:,-1]
#    learner = BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
#    t = learner.addEvidence(x,y)
#    a = learner.query(x)
#    print(a)
#    
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
    learner = BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
    x = test[:,0:-1]
    y = test[:, -1]
    test = learner.addEvidence(x, y)
    #print(learner.trees)
    a=learner.query(x)
    print(a)