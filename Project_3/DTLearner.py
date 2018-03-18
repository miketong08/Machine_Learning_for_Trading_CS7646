"""Decision Tree Learner
Python 3.6
CS7646 Project 3
Mike Tong (mtong31)
"""
import pandas as pd
import numpy as np

class DTLearner(object):
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
            best_feature = self.highest_correlation(data)
            j = data[best_feature] # column for best_feature
            split_val = j.median()


            if split_val == 0.0: # fix for median not being precise enough when values are < 1/1000's, update pandas precision
                split_val = j.min()
            
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

#import os
#os.chdir('/media/mike/9144982933/Project 3/data')
#test = pd.read_csv('Istanbul.csv', index_col='date')
#test = test.rename(columns={x:y for x,y in zip(test.columns, range(len(test.columns)))})
#test = np.array(test)
#x = test[:,:-1]
#y = test[:,-1]
#learner = DTLearner(1)
#t = learner.addEvidence(x,y)
#a = learner.query(x)
#print(a)
#learner = DTLearner(1)

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

learner = DTLearner(1)
x = test[:,0:-1]
y = test[:, -1]
test = learner.addEvidence(x, y)
print(learner.tree)
a=learner.query(x)
print(a)