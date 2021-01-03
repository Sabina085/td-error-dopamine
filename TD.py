'''
Code based on the MATLAB implementation from: https://github.com/sjgershm/RL-tutorial
'''
import numpy as np

class learning_params:
    def __init__(self, alpha, g):
        self.alpha = alpha  # learning rate
        self.g = g          # discount factor

class model_data:
    def __init__(self, w, dt, V):
        self.w = w          # weights' values
        self.dt = dt        # td-error
        self.V = V          # value function


def TD(X, r, param):
    '''
    Temporal-difference learning model
    '''
    N, D = X.shape[0], X.shape[1]
    w = np.zeros((D,1))                                  
    X = np.concatenate((X, np.zeros((1,D))), axis=0)  

    if not param:
        param = learning_params(0.1, 1)

    alpha = param.alpha    
    g = param.g            
    model = []

    for n in range(N):
        V_1 = np.dot(w.T, np.expand_dims(X[n,:], 1))            

        if n % D == 0:
            V_2 = 0                                             
        else:
            V_2 =  np.dot(w.T, np.expand_dims(X[n + 1, :], 1))  
        
        dt = r[n] +  g * V_2 - V_1                             
        w = w + alpha * dt * np.expand_dims(X[n, :], 1)          
        model.append(model_data(w, dt, V_1))
    
    return model