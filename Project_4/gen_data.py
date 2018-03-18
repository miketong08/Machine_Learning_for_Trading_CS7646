"""Defeat learners
Python 3.6
CS7646 Project 4
Mike Tong (mtong31)
"""

import numpy as np

def best4LinReg(seed=1489683273):
    """Returns input and output values that are better suited for a Linear Regression learner"""
    np.random.seed(seed)

    # dataset dimension determined by project specifications
    cols = np.random.randint(2, 1000 + 1) 
    rows = np.random.randint(10, 1000 + 1)
    
    # effictively just returns a line with some noise    
    X = np.random.random(size = (rows,cols))*200-100
    Y = np.sum(X, axis=1)
    noise = np.random.random(Y.shape) * 20
    Y_noise = Y + noise
    
    return X, Y_noise

def best4DT(seed=1489683273):
    """Returns input and output values that are better suited for Decision Tree learners"""
    np.random.seed(seed)
    
    # dataset dimension determined by project specifications
    cols = np.random.randint(2, 1000 + 1)
    rows = np.random.randint(10, 1000 + 1)
    
    # effictively just returns non-linear data
    X = np.random.random(size = (rows,cols))*200-100
    Y = np.sum(np.power(X,2), axis = 1)
    
    return X, Y

def author():
    return 'mtong31'

#if __name__=="__main__":
#    print(best4LinReg())
#    x, y = best4LinReg()
#    test = np.append(x, y, axis=1)
#    from LinRegLearner import LinRegLearner as lrl
#    from DTLearner import DTLearner
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
#    learner = lrl(1)
#    x = test[:,0:-1]
#    y = test[:, -1]
#    test = learner.addEvidence(x, y)
##    print(learner.model_coefs)
#    print(y)
#    a=learner.query(x)
#    print(a)

    
