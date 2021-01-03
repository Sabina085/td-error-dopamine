'''
Code based on the MATLAB implementation from: https://github.com/sjgershm/RL-tutorial
'''
import numpy as np

def construct_stimulus(stimulus):
    '''
    Function used for coding the predictive stimulus and the reward stimulus.
    '''    
    s = np.zeros((stimulus.trial_length, 1))
    s[stimulus.onset - 1: (stimulus.onset + stimulus.dur -1), 0] = 1
    
    return s
    
    