import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm
import logging
logger = logging.getLogger("ABM") # Get logger 

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
        self.CheckDMHasNotBeenDone = True
        
        #--- Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            if k == "InitDiv":
                self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0, infer_datetime_format = True)
            else:
                self.ObvDf[k] = pd.read_csv(v, index_col=0)
        
        # Load FlowTarget ad a DF with year in index column. Note that year should include StartYear - 1.
        # To avoid df.loc, which took a long time to get the value, we turn it into a dictionary.
        FlowTarget = pd.read_csv(self.Inputs["FlowTarget"], index_col=0)
        self.FlowTarget = FlowTarget.to_dict('dict')["FlowTarget"]
        
        # To avoid df.loc
        self.TotalDamS = {"Index": self.ObvDf["TotalDamS"].index,
                          "Value": self.ObvDf["TotalDamS"].to_numpy().flatten()}
        
        self.CCurves = self.ObvDf["CharacteristicCurves"].to_numpy()
        
        #--- Storage
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")       
        
        # Store daily output in the form of DataFrame.
        Output = pd.DataFrame(index=self.rng, columns=["DailyAction", "RequestDiv", "ActualDiv"])
        self.Output = Output.to_dict('list')    # Turn df into a dict of list form to avoid df.loc, which is slow.
        
        # Mid-calculation results.
        self.MidResult = {"RemainMonthlyDiv": 0,
                          "MonthlyDivShortage": []} 
        # For RL all variables have initial value as assigned below
        self.RL = {"Ravg": [0.5],  
                   "y": [0],
                   "R": [0],
                   "Gradient": [0],
                   "Delta": [0],
                   "Value": [1],
                   "Action": [0],
                   "Mu": [0],
                   "Mu_trunc": [0],
                   "Sig_trunc": [0],
                   "c": [0.5],
                   "YDivReq": [self.Inputs["InitYDivRef"]]} 
        
        #--- Check whether to run the DM or assigned values.
        if self.Inputs["DMFreq"] is None:
            self.AssignValue = True
            # One time assignment df.loc is fine.
            self.Output["DailyAction"][:] = self.ObvDf["AssignedBehavior"].loc[self.rng, self.Name]    
        else:
            self.AssignValue = False
            # Load initial assigned daily diversion.
            self.Output["DailyAction"][:180] = self.ObvDf["InitDiv"].loc[self.rng[:180]  , self.Name]   
            
        logger.info("Initialize irrigation diversion agent: {}".format(self.Name))
        
    def act(self, Q, AgentDict, node, CurrentDate, t):
        self.Q = Q
        self.AgentDict = AgentDict
        self.CurrentDate = CurrentDate
        self.t = t
        
        # For now we hard code the decision period here.
        if self.AssignValue is False: 
            if self.CurrentDate.month == 3 and self.CurrentDate.day == 1:
                if self.CheckDMHasNotBeenDone:
                    self.DMFunc()
                    self.CheckDMHasNotBeenDone = False
            else:
                self.CheckDMHasNotBeenDone = True
        
        if CurrentDate.month != self.t_pre_month:
            self.MidResult["MonthlyDivShortage"].append(self.MidResult["RemainMonthlyDiv"])
            self.MidResult["RemainMonthlyDiv"] = 0
        self.t_pre_month = CurrentDate.month
        RemainMonthlyDiv = self.MidResult["RemainMonthlyDiv"]
        
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]
        
        # Diversion
        if Factor < 0:
            #!!!!!!!!!!!!!! Need to be super careful!!!
            RequestDiv = -Factor*self.Output["DailyAction"][t] + RemainMonthlyDiv
            Qorg = self.Q[node][t]
            MinFlowTarget = 0   # cms
            
            if Qorg <= MinFlowTarget:
                ActualDiv = 0
                RemainMonthlyDiv = RequestDiv
                Qt = Qorg
            else:
                Qt = max(Qorg - RequestDiv, MinFlowTarget)
                ActualDiv = Qorg - Qt   # >= 0
                RemainMonthlyDiv = max(RequestDiv-ActualDiv, 0)
            
            self.Output["RequestDiv"][t] = RequestDiv
            self.Output["ActualDiv"][t] = ActualDiv
            self.MidResult["RemainMonthlyDiv"] = RemainMonthlyDiv
            self.Q[node][t] = Qt
            return self.Q
        
        elif Factor > 0:
            # Assume that diversion has beed done in t.
            Div_t = self.Output["ActualDiv"][t]
            self.Q[node][t] = self.Q[node][t] + Factor * Div_t
            return self.Q
    
    def DMFunc(self):
        Inputs = self.Inputs
        Pars = self.Pars
        RL = self.RL
        CurrentDate = self.CurrentDate
        t = self.t
        DataLength = self.DataLength
        
        FlowTarget = self.FlowTarget[CurrentDate.year - 1] 
        L = Inputs["L"]
        Lr_c = Pars["Lr_c"]
        sig = Pars["Sig"]
        b = Pars["b"]
        c = RL["c"][-1]
        Rmax = Pars["Rmax"]
        
        #--- Update
        # We use the 7 8 9 average flow (of last year), y, to calculate the deviation!
        # Maybe we can use a common pool for sharing info among agents.
        if CurrentDate.year == self.StartDate.year:
            y = FlowTarget   # Initial value (No deviation from the flow target.)
        else:
            mask = [True if i.month in [7,8,9] and (i.year == CurrentDate.year - 1) else False for i in self.rng]
            y = np.mean(self.Q["G"][mask])
        
        # Get strength
        if y >= FlowTarget: # increase div decrease c
            V = - (1 - self.getValue(y, FlowTarget, L, b))
            c = c + Lr_c * V * c
        else:
            V = (1 - self.getValue(y, FlowTarget, L, b))
            c = c + Lr_c * V * (1-c)
        
        # save
        RL["y"].append(y)
        RL["c"].append(c)       #
        RL["Value"].append(V)   
        
        #==================================================
        # Total 5 reservoirs' storage at 3/31 each year. 
        Index = np.where(self.TotalDamS["Index"] == CurrentDate.year)[0][0] 
        TotalDamS = self.TotalDamS["Value"]
        Samples = TotalDamS[max(Index-30, 0):Index]     # Use past 30 year's data to build CDF.
        x = TotalDamS[Index]               # Get predict storage at 3/31 current year. (DM is on 3/1.)
        alpha = Pars["alpha"]
        beta = Pars["beta"]
        
        #--- Get action
        q = self.genQuantile(Samples, x)        # Inverse normal CDF.
        mu = self.getMu(q, c, alpha, beta)      # Prospect function with moving center.
        action = self.getPolicyAction(mu, sig, low = -Rmax, high = Rmax)    # Truncated normal.
        #action = action*Rmax
        # save
        RL["Action"].append(action)
        RL["Mu"].append(mu)
        self.RL = RL
        
        #==================================================
        # All diversion units are in cms.
        # Calculate annual diversion request.
        CCurves = self.CCurves
        YDivRef = self.RL["YDivReq"][-1]
        YDiv = YDivRef * (1 + action)
        self.RL["YDivReq"].append(YDiv)

        #--- Map back to daily diversion 
        MRatio = np.array([IrrDiv_AgType.getMonthlyDiv(YDiv, *CCurves[m-1]) for m in [3,4,5,6,7,8,9,10,11,12,1,2]])
        MRatio = MRatio/sum(MRatio)
        MDiv = YDiv * 12 * MRatio
        
        # To daily. Uniformly assign those monthly average diversion to each day.
        if (self.CurrentDate.year + 1)%4 == 0:
            DayInMonth = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29]   # From Mar to Feb
            DDiv = []
            for m in range(12):
                DDiv += [MDiv[m]]*DayInMonth[m]
            NumDay = 366
        else:
            DayInMonth = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28]   # From Mar to Feb
            DDiv = []
            for m in range(12):
                DDiv += [MDiv[m]]*DayInMonth[m]
            NumDay = 365
        
        # Store into dataframe. 
        if DataLength - t > 366:
            self.Output["DailyAction"][t:t+NumDay] = DDiv
        else:     # For last year~~~
            self.Output["DailyAction"][t:] = DDiv[:len(self.Output["DailyAction"][t:] )]
        
        
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
        # q in [0, 1]
        # c in [0, 1]
        # mu in [-1, 1]
        # Prospect function with moving center (split point).
        if q >= c:
            mu = ((q-c)/(1-c))**alpha * (1-c) + c
        elif q < c:
            mu = -abs((q-c)/c)**beta * c + c
        # Scale to Rmax
        Rmax = self.Pars["Rmax"]
        return (mu - c)*2*Rmax

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
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        low, high = (low - mu) / sig, (high - mu) / sig
        #print(mu, "  ", low, "  ", high)
        action = truncnorm.rvs(low, high, loc=mu, scale=sig, size=1)[0]
        Mu_trunc = truncnorm.mean(low, high, loc=mu, scale=sig)
        Sig_trunc = truncnorm.std(low, high, loc=mu, scale=sig)
        
        self.RL["Mu_trunc"].append(Mu_trunc)
        self.RL["Sig_trunc"].append(Sig_trunc)
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
        b = int(round(b, 0))
        return 1 / ( 1 + ( (y - FlowTarget) / L )**(2 * b) )


    
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
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")       
        # To avoid df.loc
        self.AssignedBehavior = self.ObvDf["AssignedBehavior"].loc[self.rng, self.Name].to_numpy().flatten()
            
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
            Res_t = self.AssignedBehavior[t]
            self.Q[node][t] = Res_t
            return self.Q
            