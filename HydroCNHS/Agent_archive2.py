import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm

# This only for assigned values!!

class IrrDiv_AgType(object):
    def __init__(self, Name, Config, StartDate, DataLength):
        self.Name = Name                    # Agent name.   
        self.StartDate = StartDate          # Datetime object.
        self.DataLength = DataLength
        self.Inputs = Config["Inputs"]
        self.Attributions = Config.get("Attributions")
        self.Pars = Config["Pars"]
        
        self.CurrentDate = None             # Datetime object.
        self.t = None                       # Current time step index.
        self.t_pre = None                   # Record last t that "act()" is called,
        self.t_pre_month = StartDate.month  # Record last t month (for RemainMonthlyDiv)
        self.Q = None                       # Input outlets' flows.
        
        #--- Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0, infer_datetime_format = True)
        #if self.AssignValue:
            # Expect to be a df.
        self.AssignedBehavior = self.ObvDf["AssignedBehavior"]    
            
        # Storage
        self.TempResult = {"RemainMonthlyDiv": 0}
        self.Result = {"MonthlyDivShortage": [],
                       "ActualDiv": [],
                       "RequestDiv": []} 
        self.RL = {"Ravg": [0.5],
                   "Value": [1],
                   "Action": [0],
                   "Mu": [0],
                   "c": [0]} 
    
    def act(self, Q, AgentDict, node, CurrentDate, t):
        self.Q = Q
        self.AgentDict = AgentDict
        self.CurrentDate = CurrentDate
        self.t = t
        
        if CurrentDate.month != self.t_pre_month:
            self.Result["MonthlyDivShortage"].append(self.TempResult["RemainMonthlyDiv"])
            self.TempResult["RemainMonthlyDiv"] = 0
        self.t_pre_month = CurrentDate.month
        RemainMonthlyDiv = self.TempResult["RemainMonthlyDiv"]
        
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]
        
        # Diversion
        if Factor < 0:
            #!!!!!!!!!!!!!! Need to be super careful!!!
            RequestDiv = -Factor*self.AssignedBehavior.loc[CurrentDate, self.Name] + RemainMonthlyDiv
            Qorg = self.Q[node][self.t]
            MinFlowTarget = 0   # cms
            
            if Qorg <= MinFlowTarget:
                ActualDiv = 0
                RemainMonthlyDiv = RequestDiv
                Qt = Qorg
            else:
                Qt = max(Qorg - RequestDiv, MinFlowTarget)
                ActualDiv = Qorg - Qt   # >= 0
                RemainMonthlyDiv = max(RequestDiv-ActualDiv, 0)
            
            self.Result["RequestDiv"].append(RequestDiv)
            self.Result["ActualDiv"].append(ActualDiv)
            self.TempResult["RemainMonthlyDiv"] = RemainMonthlyDiv
            self.Q[node][self.t] = Qt
            return self.Q
        
        elif Factor > 0:
            # Assume that diversion has beed done in t.
            Div_t = self.Result["ActualDiv"][-1]
            self.Q[node][self.t] = self.Q[node][self.t] + Factor * Div_t
            return self.Q
    
    def DMFunc(self):
        Inputs = self.Inputs
        Pars = self.Pars
        RL = self.RL
        
        FlowTarget = Inputs["FlowTarget"] # Load from how to assign the value.
        L = Inputs["L"]
        Ravg = RL["Ravg"][-1]
        V_pre = RL["Value"][-1]
        Lr_Ravg = Pars["Lr_Ravg"]
        Lr_c = Pars["Lr_c"]
        action_pre = RL["Action"][-1]
        mu_pre = RL["Mu"][-1]
        sig = Pars["Sig"]
        c = RL["c"][-1]
        
        #--- Update
        y = 0 # Last year 8 9 flow deviation. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        R = self.getReward(y, FlowTarget, L)
        delta = R-Ravg + self.getValue(y, FlowTarget, L, b = 2) - V_pre
        gradient = (action_pre - mu_pre)/sig**2
        # update
        Ravg = Ravg + Lr_Ravg*delta
        c = c + Lr_c * delta * gradient
        # save
        RL["Ravg"].append(Ravg)
        RL["c"].append(c)
        
        #==================================================
        
        TotalDamS = self.ObvDf["TotalDamS"][:,self.CurrentDate] # Total 5 reservoirs' storage at 3/31 each year. 
        Samples = TotalDamS.iloc[-31:-1, 0]     # Use past 30 year's data to build CDF.
        x = TotalDamS.iloc[-1, 0]               # Get predict storage at 3/31 current year. (DM is on 3/1.)
        alpha = Pars["alpha"]
        beta = Pars["beta"]
        
        #--- Get action
        q = self.genQuantile(Samples, x)
        mu = self.getMu(q, c, alpha, beta)
        action = self.getPolicyAction(mu, sig, low = -0.9, high = 1)
        # save
        RL["Action"].append(action)
        RL["Mu"][-1].append(mu)
        
        #==================================================
        # All diversion units are in cms.
        CCurves = self.ObvDf["CharacteristicCurves"]
        if self.Result["RequestDiv"] == []:
            YDivRef = Inputs["InitYDivRef"]
        else:
            YDivRef = sum(self.Result["RequestDiv"][-365:])/365
        YDiv = YDivRef * (1 + action)
        #--- Map back to daily diversion 
        MRatio = np.array([IrrDiv_AgType.getMonthlyDiv(YDiv, *CCurves.loc[m,:]) for m in range(1, 13)])
        MRatio = MRatio/sum(MRatio)
        MDiv = YDiv * 12 * MRatio
        
        # To daily
        # How to store ????
        
        
        
    def genQuantile(self, Samples, x):
        """Calculate the quantile of x by fitting Samples with normal distribution.

        Args:
            Samples (Array): 1D array for fitting a Gaussian CDF.
            x (float): Input x for getting quentile from a Gaussian CDF.

        Returns:
            [float]: Quentile
        """
        Samples = np.array(Samples)
        Samples = Samples[~np.isnan(Samples)]
        mu, std = norm.fit(Samples)
        quantile = norm.cdf(x, loc=mu, scale=std)
        return quantile

    def getMu(self, q, c, alpha, beta):
        """A prospect function is our Mu function.

        Args:
            q (float): Quantile [0,1]
            c (float): Center (spliting point) (0, 1)
            alpha (float): Prospect function parameter.
            beta (float): Prospect function parameter.

        Returns:
            float: mu
        """
        # Prospect function with moving center (split point).
        if q >= c:
            mu = ((q-c)/(1-c))**alpha * (1-c) + c
        elif q < c:
            mu = ((q-c)/c)**beta * c + c
        return mu

    def getPolicyAction(self, mu, sig, low = -1, high = 1):
        """Draw action from a truncated Guassian distribution (policy function).

        Args:
            mu (float): Mean of truncated Guassian distribution.
            sig (float): Standard deviation of truncated Guassian distribution.
            low (int, optional): [description]. Defaults to -1.
            high (int, optional): [description]. Defaults to 1.

        Returns:
            action: The ratio for adjusting annual diversion is bounded by [low, high]. Default [-1,1].
        """
        action = truncnorm.rvs(low, high, loc=mu, scale=sig, size=1)
        return action

    def getValue(self, y, FlowTarget, L, b = 2):
        """ Get value from value function, which is a expected reward at the given state y.
            https://www.researchgate.net/figure/Three-parameters-bell-shaped-membership-function-a-b-and-c_fig2_324809710
        Args:
            y (float): Flow deviation.
            FlowTarget (float): Flow target.
            L (float): Allowed flow deviation from the flow target.
            b (float): How sharp of the drop. Default 2.
        Returns:
            float: value
        """
        return 1 / ( 1 + ( (y - FlowTarget) / L )**(2 * b) )

    def getReward(self, y, FlowTarget, L):
        """Reward function

        Args:
            y (float): Flow deviation.
            FlowTarget (float): Flow target.
            L (float): Allowed flow deviation from the flow target. 

        Returns:
            [type]: [description]
        """
        deviation = abs(y-FlowTarget)
        if deviation > L:
            Reward = 0
        else:
            Reward = 1
        return Reward
    
    @staticmethod
    def getMonthlyDiv(YDiv, a, b, LB, UB):
        if YDiv <= LB:
            MDiv = b
        elif YDiv >= UB:
            MDiv = a*(UB-LB) + b
        else:
            MDiv = a*(YDiv-LB) + b
        return MDiv
    
    
            
class ResDam_AgType(object):
    def __init__(self, Name, Config, StartDate, DataLength):
        self.Name = Name                    # Agent name.   
        self.StartDate = StartDate          # Datetime object.
        self.DataLength = DataLength
        self.Inputs = Config["Inputs"]
        self.Attributions = Config.get("Attributions")
        self.Pars = Config["Pars"]
        
        self.CurrentDate = None             # Datetime object.
        self.t = None                       # Current time step index.
        self.Q = None                       # Input outlets' flows.
        
        #--- Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0, infer_datetime_format = True)
        #if self.AssignValue:
            # Expect to be a df.
        self.AssignedBehavior = self.ObvDf["AssignedBehavior"]    
            
    def act(self, Q, AgentDict, node, CurrentDate, t):
        self.Q = Q
        self.AgentDict = AgentDict
        self.CurrentDate = CurrentDate
        self.t = t
    
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]
        
        # Release (Factor should be 1)
        if Factor < 0:
            print("Something is not right in ResDam agent.")
        
        elif Factor > 0:
            # Assume that diversion has beed done in t.
            Res_t = self.AssignedBehavior.loc[CurrentDate, self.Name]
            self.Q[node][self.t] = Res_t
            return self.Q
            