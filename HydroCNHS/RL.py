# Reinforcement learning algorithm.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import numpy as np 
from scipy.stats import norm

class Value(object):
    def __init__(self, ApproxFunc):
        self.Func = ApproxFunc
    
    def __call__(self, X, W, Return = "value", **kwargs):
        X = np.reshape(X, (-1,1))
        W = np.reshape(W, (-1,1))
        return self.Func(X, W, Return, **kwargs)
    
    @staticmethod
    def Linear(X, W, Return = "value", **kwargs):
        if Return == "value":
            return np.dot(W.T, X)[0,0]  # A single value
        elif Return == "gradient":
            return X

class Policy(object):
    def __init__(self, ApproxFunc):
        self.Func = ApproxFunc
    
    def __call__(self, X, Theta, Return = "value", **kwargs):
        X = np.reshape(X, (-1,1))
        Theta = np.reshape(Theta, (-1,1))
        return self.Func(X, Theta, Return, **kwargs)
    
    @staticmethod
    def Linear(X, Theta, Return = "value"):
        if Return == "value":
            return np.dot(Theta.T, X)[0,0]  # A single value
        elif Return == "gradient":
            return X
    
    @staticmethod
    def Gaussian(X, Theta, Return = "value", **kwargs):
        # If muIndexInfo and sigIndexInfo are not given, 
        # then we assume first half is for mu and second half is for sigma.
        if kwargs.get("muIndexInfo") is None and kwargs.get("sigIndexInfo") is None:
            half = int(Theta.shape[0]/2)
            muII = [0, half]
            sigII = [half, Theta.shape[0]]
        else:
            muII = kwargs["muIndexInfo"]     # muIndexInfo = [StartIndex, length]
            sigII = kwargs["sigIndexInfo"]   # sigIndexInfo = [StartIndex, length]
        
        if Return == "value":
            # Get functions that are used to calculate mu and sig from X and Theta.
            if kwargs.get("muFunc") is None:
                raise KeyError("muFunc argument is missing.")
            if kwargs.get("sigFunc") is None:
                raise KeyError("sigFunc argument is missing.")
            muFunc = kwargs["muFunc"]
            sigFunc = kwargs["sigFunc"]
            
            # Calculate mu and sig using approximation functions.
            mu = muFunc(X[muII[0]:muII[0]+muII[1], :], Theta[muII[0]:muII[0]+muII[1], :])
            sig = sigFunc(X[sigII[0]:sigII[0]+sigII[1], :], Theta[sigII[0]:sigII[0]+sigII[1], :])
            
            # Generate action according a normal distribution.
            rn = np.random.uniform()
            action = norm.ppf(rn, loc = mu, scale = sig)
            
            return (action, mu, sig)
        
        elif Return == "gradient":
            if kwargs.get("actionTuple") is None:
                raise KeyError("actionTuple (action, mu, sig) argument is missing.")
        
            a, mu, sig = kwargs["actionTuple"] 
            Xmu = X[muII[0]:muII[0]+muII[1], :]
            Xsig = X[sigII[0]:sigII[0]+sigII[1], :]
            
            dmu = 1/sig**2 * (a-mu) * Xmu
            dsig = ( ((a-mu)**2 / sig**2 - 1) * Xsig )
            if muII[0] < sigII[0]:
                gradient = np.concatenate((dmu, dsig), axis=0)
            else:
                gradient = np.concatenate((dsig, dmu), axis=0)
            
            return gradient


class Actor_Critic(object):
    def __init__(self, ValueFunc, PolicyFunc, Pars, **kwargs):
        # Assign value and policy approximation function.
        # ====================================================================
        # Note that the reward function is defined outside of this class. 
        # Actor_Critic class will only take in reward R immediately in 
        # updatePars().
        # ====================================================================
        self.Value = ValueFunc
        self.Policy = PolicyFunc

        # Load parameter from Pars
        # ====================================================================
        # Note that this Actor_Critic is design for linear approximation 
        # functions, which X and Theta are all 1D array with shape = (-1, 1).
        # For nonlinear (e.g. neural network), we need to modify code to allow
        # multi-layer of W and Theta in a form of dictionaries. (See ML proj).     
        # ====================================================================
        self.W = np.array(Pars["W"]).reshape((-1,1))
        self.Theta = np.array(Pars["Theta"]).reshape((-1,1))
        self.LR_W = np.array(Pars["LR_W"]).reshape((-1,1))
        self.LR_T = np.array(Pars["LR_T"]).reshape((-1,1))
        self.LR_R = Pars["LR_R"]
        self.Lambda_W = Pars["Lambda_W"]
        self.Lambda_T = Pars["Lambda_T"]
        
        # Initialize variables
        self.AvgR = 0                           # Average rewards
        self.Z_W = np.zeros(self.W.shape)       # Eligibility trace of W
        self.Z_T = np.zeros(self.Theta.shape)   # Eligibility trace of Theta
        self.PreX = None                        # Previous state feature vector
        
        # Collect other kwargs
        self.kwargs = kwargs
    
    def getAction(self, X):
        X = np.reshape(X, (-1,1))
        Theta = self.Theta
        kwargs = self.kwargs
        actionTuple = self.Policy(X, Theta, **kwargs)
        self.PreX = X
        return actionTuple
    
    def updatePars(self, X_new, R, **kwargs):
        V = self.Value
        P = self.Policy
        Theta = self.Theta
        W = self.W
        kwargs = {**self.kwargs, **kwargs}      # Merge two kwargs dictionaries.
        
        # Update parameters
        delta = R - self.AvgR + V(X_new, W, **kwargs) - V(self.PreX, W, **kwargs)
        self.AvgR = self.AvgR + self.LR_R*delta
        self.Z_W = self.Lambda_W * self.Z_W + V(self.PreX, W, "gradient", **kwargs)
        self.Z_T = self.Lambda_T * self.Z_T + P(self.PreX, Theta, "gradient", **kwargs)
        self.W = W + self.LR_W * (self.Z_W * delta)
        self.Theta = Theta + self.LR_T * (self.Z_T * delta)