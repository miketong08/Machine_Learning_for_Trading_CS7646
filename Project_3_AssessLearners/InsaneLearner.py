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
