"""
CS7646 ML For Trading
Project 7: Q-Learner
Michael Tong (mtong31)

This script provides the QLearner class which generates and updates a Q table with Dyna-Q functionality.
"""

import numpy as np

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros(shape=(num_states, num_actions))
        
        if dyna != 0:
            self.T_c = np.zeros(shape=(num_states, num_actions, num_states))
            self.T_c.fill(0.0001) # small value is to prevent divide by zero errors
            self.T = self.T_c/np.sum(self.T_c, axis=2, keepdims=True)
            
            self.R = self.Q.copy()
            self.R.fill(-1.0) # piazza solution, not sure why this works
    
    def get_Q(self):
        return(self.Q)
        
    def get_T(self):
        return(self.T)
        
    def get_R(self):
        return(self.R)
        
    def update_Q(self,s,a,s_prime,r):
        """Updates the Q Table based on the new state and reward of the query, returns a Q Table"""
        return((1-self.alpha) * self.Q[s][a] + self.alpha*(r + self.gamma * self.Q[s_prime][np.argmax(self.Q[s_prime])]))
    
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.random.randint(0, self.Q.shape[1])
        if self.verbose: print("s =", s,"a =",action)
        return action
    
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The reward
        @returns: The selected action
        """
        s = self.s
        a = self.a
        num_states = self.Q.shape[0]
        num_actions = self.Q.shape[1]
        
        self.Q[s][a] = self.update_Q(s, a, s_prime, r)
        
        if np.random.randint(0,101)/100 > self.rar:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(self.Q[s_prime])
        self.rar *= self.radr
        
        if self.dyna != 0:
            self.T_c[s][a][s_prime] += 1
            self.T = self.T_c/np.sum(self.T_c, axis=2, keepdims=True)
            self.R[s][a] = (1-self.alpha) * self.R[s][a] + (self.alpha * r)
            
            for _ in range(self.dyna):
                s_dyna  = np.random.randint(0, num_states)                
                a_dyna  = np.random.randint(0, num_actions)
                s_prime_dyna = np.random.multinomial(1, self.T[s_dyna][a_dyna]).argmax()
                r_dyna = self.R[s_dyna][a_dyna]
                self.Q[s_dyna][a_dyna] = self.update_Q(s_dyna, a_dyna, s_prime_dyna, r_dyna)

        self.s = s_prime
        self.a = action
        
        if self.verbose: print("s =", s_prime,"a =", action ,"r =",r)
        return action
        
    def author(self):
        return('mtong31')
        
if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
