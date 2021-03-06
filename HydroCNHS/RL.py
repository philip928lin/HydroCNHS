# Reinforcement learning algorithm.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

import numpy as np 
from scipy.stats import norm

class Value(object):
    """Approximate value function class for Actor_Critic.
        Run Value.getAvailableFunctions() to see available functions.
    """
    def __init__(self, ApproxFunc):
        """Approximate value function for Actor_Critic.

        Args:
            ApproxFunc (function/str): Approximate value function. EX Value.Linear
        """
        if isinstance(ApproxFunc, str):
            self.Func = eval(ApproxFunc)    # Parse string into a function object.
        else:
            self.Func = ApproxFunc          # Directly assign a function object.
    
    def __call__(self, X, W, Return = "value", **kwargs):
        """Return value or gradient of approximate value function.

        Args:
            X (array): Input feature vector.
            W (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of value function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        X = np.reshape(X, (-1,1))
        W = np.reshape(W, (-1,1))
        return self.Func(X, W, Return, **kwargs)
    
    @staticmethod
    def getAvailableFunctions():
        """Get available approximate value functions.

        Returns:
            list: List of available approximate value functions.
        """
        FuncList = [func for func in dir(Value) if callable(getattr(Value, func)) and not func.startswith("__")]
        FuncList.remove("getAvailableFunctions")
        print("Approximate value functions: {}".format(FuncList))
        return FuncList
    
    @staticmethod
    def Linear(X, W, Return = "value", **kwargs):
        """Linear = W^TX

        Args:
            X (array): Input feature vector.
            W (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of value function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        if Return == "value":
            return np.dot(W.T, X)[0,0]  # A single value
        elif Return == "gradient":
            return X
    
    @staticmethod
    def Sigmoid(X, W, Return = "value", **kwargs):
        """Sigmoid = Sigmoid(W^TX)

        Args:
            X (array): Input feature vector.
            W (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of value function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        Z = Value.Linear(X, W, **kwargs)
        if Z < 0:
            A = 1 - 1 / (1 + np.exp(Z))
        elif Z >= 0:
            A = 1 / (1 + np.exp(-Z))
        
        if Return == "value":
            return A
        
        elif Return == "gradient":
            # Apply chain rule and back to calculate the gradient for W.
            # df(z(W^TX)=Z) / dW  =  df/dZ * dZ/dW
            df = A * (1 - A)
            dZ = Value.Linear(X, W, "gradient", **kwargs)
            return df * dZ
        
    @staticmethod
    def Tanh(X, W, Return = "value", **kwargs):
        """Tanh = Tanh(W^TX)

        Args:
            X (array): Input feature vector.
            W (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of value function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        Z = Value.Linear(X, W, **kwargs)
        expZ = np.exp(Z)
        exp_Z = np.exp(-Z)
        A = (expZ - exp_Z) / (expZ + exp_Z)

        if Return == "value":
            return A
        
        elif Return == "gradient":
            # Apply chain rule and back to calculate the gradient for W.
            # df(z(W^TX)=Z) / dW  =  df/dZ * dZ/dW
            df = 1 - A**2
            dZ = Value.Linear(X, W, "gradient", **kwargs)
            return df * dZ

class Policy(object):
    """Approximate policy function class for Actor_Critic.
        Run Policy.getAvailableFunctions() to see available functions.
    """
    def __init__(self, ApproxFunc):
        """Approximate policy function for Actor_Critic.

        Args:
            ApproxFunc (function/str): Approximate value function. EX Policy.Gaussian
        """
        if isinstance(ApproxFunc, str):
            self.Func = eval(ApproxFunc)    # Parse string into a function object.
        else:
            self.Func = ApproxFunc          # Directly assign a function object.
    
    def __call__(self, X, Theta, Return = "value", **kwargs):
        """Return value or gradient of approximate policy function.

        Args:
            X (array): Input feature vector.
            Theta (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of policy function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        X = np.reshape(X, (-1,1))
        Theta = np.reshape(Theta, (-1,1))
        return self.Func(X, Theta, Return, **kwargs)
    
    @staticmethod
    def getAvailableFunctions():
        """Get available approximate policy functions.

        Returns:
            list: List of available approximate policy functions.
        """
        FuncList = ["Gaussian"]
        FuncList2 = [func for func in dir(Policy) if callable(getattr(Policy, func)) and not func.startswith("__") and func not in FuncList ]
        FuncList2.remove("getAvailableFunctions")
        print("Approximate policy functions: {}".format(FuncList))
        print("Auxiliary functions: {}".format(FuncList2))
        return FuncList
    
    @staticmethod
    def Linear(X, Theta, Return = "value", **kwargs):
        """This is an auxiliary function. Linear = Theta^TX.

        Args:
            X (array): Input feature vector.
            Theta (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of the function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        if Return == "value":
            return np.dot(Theta.T, X)[0,0]  # A single value
        elif Return == "gradient":
            return X
        
    @staticmethod
    def Sigmoid(X, Theta, Return = "value", **kwargs):
        """This is an auxiliary function. Sigmoid = Sigmoid(Theta^TX).

        Args:
            X (array): Input feature vector.
            Theta (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of the function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        Z = Value.Linear(X, Theta, **kwargs)
        if Z < 0:
            A = 1 - 1 / (1 + np.exp(Z))
        elif Z >= 0:
            A = 1 / (1 + np.exp(-Z))
        
        if Return == "value":
            return A
        
        elif Return == "gradient":
            # Apply chain rule and back to calculate the gradient for Theta.
            # df(z(Theta^TX)=Z) / dW  =  df/dZ * dZ/dW
            df = A * (1 - A)
            dZ = Value.Linear(X, Theta, "gradient", **kwargs)
            return df * dZ
        
    @staticmethod
    def Tanh(X, Theta, Return = "value", **kwargs):
        """This is an auxiliary function. Tanh = Tanh(Theta^TX).

        Args:
            X (array): Input feature vector.
            Theta (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of the function. Defaults to "value".

        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        Z = Value.Linear(X, Theta, **kwargs)
        expZ = np.exp(Z)
        exp_Z = np.exp(-Z)
        A = (expZ - exp_Z) / (expZ + exp_Z)

        if Return == "value":
            return A
        
        elif Return == "gradient":
            # Apply chain rule and back to calculate the gradient for Theta.
            # df(z(Theta^TX)=Z) / dTheta  =  df/dZ * dZ/dTheta
            df = 1 - A**2
            dZ = Value.Linear(X, Theta, "gradient", **kwargs)
            return df * dZ    
        
    @staticmethod
    def Gaussian(X, Theta, Return = "value", **kwargs):
        """Gaussian = Gaussian(mu, sig). 
            mu = muFunc(Theta_mu^T dot X_mu).
            sig = sigFunc(Theta_sig^T dot X_sig).
        Args:
            X (array): Input feature vector.
            Theta (array): Input weight vector.
            Return (str, optional): "value" or "gradient" of the function. Defaults to "value".
            muIndexInfo (list, optional): [StartIndex, length]. Default [0, int(Theta.shape[0]/2)].
            sigIndexInfo (list, optional): [StartIndex, length]. Default [int(Theta.shape[0]/2, int(Theta.shape[0])].
            muFunc (function/str): Auxiliary function.
            sigFunc (function/str): Auxiliary function.
            actionTuple (tuple): (action, mu, sig). Only need it when Return = "gradient".
            
        Returns:
            float/array: "value": float; "gradient": array with size = (-1, 1).
        """
        # If muIndexInfo and sigIndexInfo are not given, 
        # then we assume first half is for mu and second half is for sigma.
        if kwargs.get("muIndexInfo") is None and kwargs.get("sigIndexInfo") is None:
            half = int(Theta.shape[0]/2)
            muII = [0, half]
            sigII = [half, Theta.shape[0]]
        else:
            muII = kwargs["muIndexInfo"]     # muIndexInfo = [StartIndex, length]
            sigII = kwargs["sigIndexInfo"]   # sigIndexInfo = [StartIndex, length]
        
        # Get auxiliary functions for mu and sig.
        if kwargs.get("muFunc") is None:
            raise KeyError("muFunc argument is missing.")
        if kwargs.get("sigFunc") is None:
            raise KeyError("sigFunc argument is missing.")
        
        def toFunc(Func):
            if isinstance(Func, str):
                return eval(Func)    # Parse string into a function object.
            else:
                return Func          # Directly assign a function object.
        muFunc = toFunc(kwargs["muFunc"])
        sigFunc = toFunc(kwargs["sigFunc"])
        
        if Return == "value":
            # Calculate mu and sig using approximation functions.
            mu = muFunc(X[muII[0]:muII[0]+muII[1], :], Theta[muII[0]:muII[0]+muII[1], :], **kwargs)
            sig = sigFunc(X[sigII[0]:sigII[0]+sigII[1], :], Theta[sigII[0]:sigII[0]+sigII[1], :], **kwargs)
            
            # Generate action according a normal distribution.
            rn = np.random.uniform()
            action = norm.ppf(rn, loc = mu, scale = sig)
            
            return (action, mu, sig)
        
        elif Return == "gradient":
            if kwargs.get("actionTuple") is None:
                raise KeyError("actionTuple (action, mu, sig) argument is missing.")
            a, mu, sig = kwargs["actionTuple"] 
            
            # Apply chain rule and back to calculate the gradient for Theta.
            # df(z(Theta^TX)=Z) / dTheta  =  df/dZ * dZ/dTheta
            dmu = 1/sig**2 * (a-mu)
            dsig = ((a-mu)**2 / sig**2 - 1)
            dZmu = muFunc(X[muII[0]:muII[0]+muII[1], :], Theta[muII[0]:muII[0]+muII[1], :], "gradient", **kwargs)
            dZsig = sigFunc(X[muII[0]:muII[0]+muII[1], :], Theta[muII[0]:muII[0]+muII[1], :], "gradient", **kwargs)
            dTmu = dmu * dZmu
            dTsig = dsig * dZsig
            
            if muII[0] < sigII[0]:
                gradient = np.concatenate((dTmu, dTsig), axis=0)
            else:
                gradient = np.concatenate((dTsig, dTmu), axis=0)
            return gradient

class Actor_Critic(object):
    def __init__(self, ValueFunc, PolicyFunc, Pars, **kwargs):
        """A general actor-critic algorithm framework for reinforcement learning. 

        Args:
            ValueFunc (function/str): Approximate value function. EX Value.Linear or "Value.Linear".
            PolicyFunc (function/str): Approximate policy function. EX Policy.Gaussian or "Policy.Gaussian".
            Pars (dict): model.yaml input parameter for actor-critic algorithm.
            **kwargs: Include arguments of other auxiliary functions or settings. Please check the requirements of given ValueFunc and PolicyFunc in Value and Policy classes.
        """
        # Assign value and policy approximation function.
        # ====================================================================
        # Note that the reward function is defined outside of this class. 
        # Actor_Critic class will only take in reward R immediately in 
        # updatePars().
        # ====================================================================
        def toFunc(Func):
            if isinstance(Func, str):
                return eval(Func)    # Parse string into a function object.
            else:
                return Func          # Directly assign a function object.
        self.Value = toFunc(ValueFunc)
        self.Policy = toFunc(PolicyFunc)

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
        self.LR_R = Pars["LR_R"][0]
        self.Lambda_W = Pars["Lambda_W"][0]
        self.Lambda_T = Pars["Lambda_T"][0]
        
        # Initialize variables
        self.AvgR = 0                           # Average rewards
        self.Z_W = np.zeros(self.W.shape)       # Eligibility trace of W
        self.Z_T = np.zeros(self.Theta.shape)   # Eligibility trace of Theta
        self.PreX = None                        # Previous state feature vector
        
        # Collect other kwargs
        self.kwargs = kwargs
        
        # Control variable
        self.Initial = True
    
    def getAction(self, X):
        """Get action according to the policy.

        Args:
            X (array): A feature vector.

        Returns:
            tuple: actionTuple.
        """
        X = np.reshape(X, (-1,1))
        Theta = self.Theta
        kwargs = self.kwargs
        actionTuple = self.Policy(X, Theta, **kwargs)
        self.PreX = X
        self.Initial = False
        return actionTuple
    
    def updatePars(self, X_new, R, **kwargs):
        """Update parameters including average reward (AvgR), eligible traces (Z_W & Z_T), and weight vectors for value and policy functions (W & Theta).

        Args:
            X_new (array): A feature vector of next state.
            R (float): Reward from taking the action in last state. We didn't provide the reward function for users. Please feed in the Reward directly.
            **kwargs: If adopted Gaussian Policy function, actionTuple needs to be given.
        """
        if self.Initial:
            # Do nothing.
            # Making sure that if there are no actions have been taken, then no parameters are updated.
            self.Initial = False
        else: 
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