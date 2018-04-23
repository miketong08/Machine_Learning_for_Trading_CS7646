"""Bag Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""

import numpy as np
import pandas as pd

class BagLearner(object):
    def __init__(self, learner, kwargs = {}, bags=20, boost=False, verbose=False):
        
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
            
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.trees = []

    def author(self):
        return('mtong31')
    
    def addEvidence(self, Xtrain, Ytrain):
        df = pd.DataFrame(Xtrain)
        df['output'] = Ytrain
        
        if self.boost:
            if 'weight' not in df:
                df['weight'] = 1./df.shape[0]
                
        for method in self.learners:
            if self.boost:
                learning_df = pd.DataFrame([df.sample(weights=df['weight']).values[0] for i in range(df.shape[0])])
                X = learning_df.iloc[:, :-2]
                Y = learning_df.iloc[:,-2]
            else:
                learning_df = pd.DataFrame([df.sample().values[0] for i in range(df.shape[0])])
                X = learning_df.iloc[:, :-1]
                Y = learning_df.iloc[:,-1]
                
            method.addEvidence(X,Y)
            self.trees.append(method.tree)

            if self.boost:
                predictedY = self.query_trees(X.values, self.trees[-1])
                
                # should consider a different error function for future iterations
                error_function = (predictedY - Y)**2 + 1 # +1 to prevent error's of 1 not impacting the weight   
                learning_df['error'] = error_function
                learning_df = learning_df[learning_df['error'] != 1].drop_duplicates()
                affected_indicies = [df.iloc[:, :-1].isin(i).all(1).nonzero()[0][0] for i in\
                                     learning_df.iloc[:, :-1].values]

                df.iloc[affected_indicies, -1] *= list(learning_df['error'])
                df['weight'] = df['weight']/df['weight'].sum()

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
        queries = np.average(queries,axis=0)
        
        return(queries)
            
            
