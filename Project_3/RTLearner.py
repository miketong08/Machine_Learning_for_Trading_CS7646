"""Random Tree Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""

import pandas as pd
import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
    
    def author(self):
        return('mtong31')
        
    def addEvidence(self, Xtrain, Ytrain):
        """Accepts inputs (Xtrain) and outputs (Ytrain) and calls the build_tree function on the data, updates the tree attribute"""
        dataframe = pd.DataFrame(Xtrain)
        dataframe['Y'] = Ytrain.reshape(len(Ytrain),1)
        
        if self.tree == None:
            self.tree = self.build_tree(dataframe)
            
#        else:
#            self.tree = np.vstack(self.tree, self.build_tree(dataframe))

#        return(self.tree)
    
    def highest_correlation(self, df):
        """Returns the highest correlated value by it's df index"""
        correlations = np.tril(np.array(df.corr()), k=-1)

        return(abs(np.nan_to_num(correlations[-1])).argmax())
    
    def build_tree(self, data): 
        """Recursively build's a tree by returning arrays in the form [feature, split value, less than index, greater than index]
        leaf values are denoted as feature == -1""" 
        if data.empty:
            return(np.array([np.nan, np.nan, np.nan, np.nan]))
        
        if self.leaf_size == 1 and data.shape[0] == 1:
            return(np.array([-1, data['Y'], np.nan, np.nan]).reshape(1,4))
        
        # returns leaf value, if size is greater than 1, will return mode of values
        elif data.shape[0] <= self.leaf_size or len(pd.unique(data.ix[:,-1])) == 1:
            leaf_values = data['Y'].mode()
            if len(leaf_values) == 1:
                leaf_val = float(leaf_values[0])
            else:
                leaf_val = float(np.random.choice(leaf_values))
            return(np.array([-1, leaf_val, np.nan, np.nan]).reshape(1,4))
        
        else:
            # randmized feature selection, averages two selections
            best_feature = np.random.choice(data.columns[:-1])
            j = data[best_feature]
            i1, i2 = np.random.choice(j,size=2)
            split_val = (float(i1) + float(i2))/2.0
            
            left_tree  = self.build_tree(data[data.ix[:, data.columns.get_loc(best_feature)] <= split_val])
            right_tree = self.build_tree(data[data.ix[:, data.columns.get_loc(best_feature)] > split_val])
            
            root = [best_feature, split_val, 1, left_tree.shape[0] + 1]   
            temp_tree = np.vstack([root, left_tree, right_tree])
            return(temp_tree)
    
    def query(self,Xtest):
        """Given an input (Xtest), returns the associated query output(s), can accept arrays"""
        results = []
        for i in Xtest:
            current_pos = 0
            while current_pos < 100000:    
                tree_pos = self.tree[current_pos]
                if int(tree_pos[0]) == -1:
                    results.append(float(tree_pos[1]))
                    break
                
                if i[int(tree_pos[0])] <= tree_pos[1]:
                    current_pos += 1
                else:
                    current_pos += int(tree_pos[3]) 
        return(results)