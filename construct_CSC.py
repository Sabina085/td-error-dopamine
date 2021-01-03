'''
Code based on the MATLAB implementation from: https://github.com/sjgershm/RL-tutorial
'''
import numpy as np

def construct_CSC(stimulus):
    '''
    Function used for constructing the complete serial compound (CSC)
    representation of the predictive stimulus.
    '''
    x = np.diag(stimulus[:, 0])
        
    for i in range(1, stimulus.shape[1]):
        x = np.concatenate((x, np.diag(stimulus[:, i])), axis=1)

    return x
    

       
