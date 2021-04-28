import os
import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm, multivariate_normal, uniform
import logging

logger = logging.getLogger("ABM") 

class DivDM(object):
    def __init__(self, StartDate, DataLength, ABM):
        BasicPath = r"C:\Users\ResearchPC\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\YRBModel\INPUT"
        self.Path = {"FlowTarget": os.path.join(BasicPath, "FlowTarget_Cali.csv"),
                     "TotalDamS": os.path.join(BasicPath, "Storage_M_Total_km2-m.csv"),
                     "CCurves": r"C:\Users\ResearchPC\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\Data",
                     "AgCor": os.path.join(BasicPath, "AgCor.csv")}
        self.AgList = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
        self.Obv = {}
        self.Ag = {}
        self.t = None
        self.t_pre = None
        self.StartDate = StartDate
        self.DataLength = DataLength
        self.SocialNorm = False
        
        #--- AgPars, AgInputs (Not optimize)
        self.AgPars = {}
        self.AgInputs = {}
        for ag in self.AgList:
            for agType in ABM:
                if ag in ABM[agType]:
                    self.AgPars[ag] = ABM[agType][ag]["Pars"]
                    self.AgInputs[ag] = ABM[agType][ag]["Input"]
        
        #--- Store space
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")
        Output = pd.DataFrame(index=self.rng, columns=["DailyAction", "RequestDiv", "ActualDiv"]).to_dict('list')
        for ag in self.AgList:
            self.Ag[ag] = {"RL":{"y": [None],
                                 "V": [0],
                                 "Vavg": [0],
                                 "q": [None],
                                 "Action": [0],
                                 "Mu": [0],
                                 "Mu_trunc": [0],
                                 "Sig_trunc": [0],
                                 "c": [0.5],
                                 "YDivReq": [ AgInputs[ag]["InitYDivRef"] ] },
                           "Output": Output}
        #--- Flowtarget
        FlowTarget = pd.read_csv(self.Path["FlowTarget"], index_col=0)
        self.FlowTarget = FlowTarget.to_dict('dict')["FlowTarget"]
        
        #--- Feature - TotalDamS
        TotalDamS = pd.read_csv(self.Path["TotalDamS"], index_col=0)
        self.TotalDamS = {"Index": TotalDamS.index,
                          "Value": TotalDamS.to_numpy().flatten()}
        #--- CCurves
        CCurves = {}
        for ag in self.AgList:
            path = os.path.join(self.Path["CCurves"], "MRatio_{}.csv".format(ag))
            CCurves[ag] = pd.read_csv(path, index_col=0).to_numpy()
        self.CCurves = CCurves
        
        #--- Corr
        if self.Path["AgCor"] is not None:
            self.Corr = pd.read_csv(self.Path["AgCor"], index_col=0).loc[self.AgList, self.AgList].to_numpy()
            self.SocialNorm = True
            
            
    def __call__(self, Q, t, CurrentDate, ag, Output, AssignValue = False):
        if AssignValue:     # Do nothing.
            return Output
        
        self.Q = Q
        self.t = t
        self.CurrentDate = CurrentDate
        self.ag = ag
        self.AG[ag]["Output"] = Output
        
        # Only run DMFunc when it has not been done at t. DMFunc will go through all agents.
        if t != self.t_pre:
            self.DMFunc()
            self.t_pre = t
        return self.AG[ag]["Output"]
    
    def DMfunc(self):
        AgList = self.AgList
        CurrentDate = self.CurrentDate
        FlowTarget = self.FlowTarget[CurrentDate.year - 1]
        
        #==================================================
        #--- Update c by BM algorithm
        if CurrentDate.year == self.StartDate.year:
            y = FlowTarget   # Initial value (No deviation from the flow target.)
        else:
            mask = [True if i.month in [7,8,9] and (i.year == CurrentDate.year - 1) else False for i in self.rng]
            y = np.mean(self.Q["G"][mask])
        
        for ag in AgList:
            RL = self.Ag[ag]["RL"]
            Pars = self.AgPars[ag]
            
            L = Pars["L"]
            Lr_c = Pars["Lr_c"]
            V_pre = RL["V"][:-9]
            
            # Sigmoid function as the value function
            V = (FlowTarget - y)/L
            if V < 0:    # To avoid overflow issue
                V = 1 - 1/(1+np.exp(V)) - 0.5
            else:
                V = 1/(1+np.exp(-V)) - 0.5   
            V = 2*V     # V in  [-1, 1]
            Vavg = np.sum([V] + V_pre)/10     # Average over past ten years
            
            # BM model      (Note: Lr_c * Vavg or V has to be in [-1, 1])     
            # Choose V or Vavg
            if V >= 0:   # Sim < Target: increase center => decrease Div 
                c = c + V*Lr_c*(1-c)
            else:        # Sim > Target: decrease center => increase Div
                c = c + V*Lr_c*c  
            
            # Save
            RL["y"].append(y)
            RL["c"].append(c)       
            RL["V"].append(V)   
            RL["Vavg"].append(Vavg) 
            self.Ag[ag]["RL"] = RL
        #==================================================
        #==================================================
        #--- Get feature
        Index = np.where(self.TotalDamS["Index"] == CurrentDate.year)[0][0] 
        TotalDamS = self.TotalDamS["Value"]
        Samples = np.array(TotalDamS[max(Index-30, 0):Index])     # Use past 30 year's data to build CDF.
        x = TotalDamS[Index]               # Get predict storage at 3/31 current.
        mu, std = norm.fit(Samples)
        q = norm.cdf(x, loc=mu, scale=std)
        #==================================================
        #==================================================
        #--- Get Mu
        for ag in AgList:
            RL = self.Ag[ag]["RL"]
            Pars = self.AgPars[ag]
            alpha = Pars["alpha"]
            beta = Pars["beta"]
            Rmax = Pars["Rmax"]
            # Prospect function
            if q >= c:
                mu = ((q-c)/(1-c))**alpha * (1-c) + c
            elif q < c:
                mu = -abs((q-c)/c)**beta * c + c
            # Scale to Rmax
            Mu = (mu - c)*2*Rmax
            # Save
            RL["q"].append(q)
            RL["Mu"].append(Mu)
            self.Ag[ag]["RL"] = RL
        #==================================================
        #==================================================
        #--- Get Action
        # Get uniform rn for quantile mapping.
        if self.SocialNorm:
            Corr = self.Corr
            rn_cor = multivariate_normal.rvs(cov = Corr, size=1)
            rn = norm.cdf(rn_cor)
        else:
            rn = uniform.rvs(size=len(AgList))
            
        # Inverse of truncated normal CDF
        for i, ag in enumerate(AgList):
            RL = self.Ag[ag]["RL"]
            Pars = self.AgPars[ag]
            Rmax = Pars["Rmax"]
            mu = RL["Mu"][-1]
            sig = Pars["Sig"]
            rn_q = rn[i]
            # Inverse CDF
            low, high = (-Rmax - mu) / sig, (Rmax - mu) / sig
            action = truncnorm.ppf(rn_q, low, high, loc=mu, scale=sig)
            Mu_trunc = truncnorm.mean(low, high, loc=mu, scale=sig)
            Sig_trunc = truncnorm.std(low, high, loc=mu, scale=sig)
            
            RL["Action"].append(action)
            RL["Mu_trunc"].append(Mu_trunc)
            RL["Sig_trunc"].append(Sig_trunc)
            self.Ag[ag]["RL"] = RL
        #==================================================
        #==================================================
        #--- Mapping back to daily diversion
        def getMonthlyDiv(YDiv, a, b, LB, UB):
            if YDiv <= LB:
                MDiv = b
            elif YDiv >= UB:
                MDiv = a*(UB-LB) + b
            else:
                MDiv = a*(YDiv-LB) + b
            return MDiv

        t = self.t
        DataLength = self.DataLength
        for ag in AgList:
            RL = self.Ag[ag]["RL"]
            CCurves = self.CCurves[ag]
            YDivRef = RL["YDivReq"][-1]
            MaxYDiv = self.AgInputs[ag]["MaxYDiv"]
            MinYDiv = self.AgInputs[ag]["MinYDiv"]
            
            YDiv = YDivRef * (1 + action)
            # Hard constraint for MaxYDiv and MinYDiv
            if MaxYDiv is not None and MinYDiv is not None:
                if YDiv > MaxYDiv:
                    YDiv = MaxYDiv
                elif YDiv < MinYDiv:
                    YDiv = MinYDiv
            self.RL["YDivReq"].append(YDiv)

            # Map back to daily diversion (from Mar to Feb)
            MRatio = np.array([getMonthlyDiv(YDiv, *CCurves[m-1]) for m in [3,4,5,6,7,8,9,10,11,12,1,2]])
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
                
class IrrDiv_AgType(object):
    def __init__(self, Name, Config, StartDate, DataLength):
        self.Name = Name                    # Agent name.   
        self.StartDate = StartDate          # Datetime object.
        self.t_pre_month = StartDate.month  # Record last t month (for RemainMonthlyDiv)
        self.DataLength = DataLength
        self.Inputs = Config["Inputs"]
        self.Attributions = Config.get("Attributions")
        self.Pars = Config["Pars"]
        
        #--- Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            if k == "InitDiv":
                self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0, infer_datetime_format = True)
            else:
                self.ObvDf[k] = pd.read_csv(v, index_col=0)
            logger.info("[{}] Load {}: {}".format(self.Name, k, v))
                
        #--- Store space
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")       
        Output = pd.DataFrame(index=self.rng, columns=["DailyAction", "RequestDiv", "ActualDiv"])
        self.Output = Output.to_dict('list')    # Turn df into a dict of list form to avoid df.loc, which is slow.
        
        # Mid-calculation results.
        self.MidResult = {"RemainMonthlyDiv": 0,
                          "MonthlyDivShortage": []} 
        
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
        
    def act(self, Q, AgentDict, node, CurrentDate, t, DM):
        self.Q = Q
        self.CurrentDate = CurrentDate
        self.t = t
        ag = self.Name
        Output = self.Output
        MidResult = self.MidResult
        
        #==================================================
        #--- Get Output (Diversion decision)
        # For now we hard code the decision period here.
        if self.AssignValue is False: 
            if self.CurrentDate.month == 3 and self.CurrentDate.day == 1:
                Output = DM(Q, t, CurrentDate, ag, Output, AssignValue = False)
        #==================================================
        #==================================================
        #--- Calculate the actual request.
        if CurrentDate.month != self.t_pre_month:
            MidResult["MonthlyDivShortage"].append(MidResult["RemainMonthlyDiv"])
            MidResult["RemainMonthlyDiv"] = 0
        self.t_pre_month = CurrentDate.month
        RemainMonthlyDiv = self.MidResult["RemainMonthlyDiv"]
        
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]
        
        # Diversion
        if Factor < 0:
            #!!!!!!!!!!!!!! Need to be super careful !!!
            RequestDiv = (-Factor * Output["DailyAction"][t]) + RemainMonthlyDiv
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
            
            Output["RequestDiv"][t] = RequestDiv
            Output["ActualDiv"][t] = ActualDiv
            MidResult["RemainMonthlyDiv"] = RemainMonthlyDiv
            self.Output = Output
            self.MidResult = MidResult
            self.Q[node][t] = Qt
            return self.Q
        
        elif Factor > 0:
            # Assume that the diversion has beed done in t.
            Div_t = self.Output["ActualDiv"][t]
            self.Q[node][t] = self.Q[node][t] + Factor * Div_t
            return self.Q
    
class IrrDiv_RWS_AgType(object):
    def __init__(self, Name, AgConfig, StartDate, DataLength):
        self.Name = Name                    # Agent name.   
        self.AgList = ['Roza', 'Wapato', 'Sunnyside']
        self.Proportion = [0.44, 0.39, 0.17]
        self.StartDate = StartDate          # Datetime object.
        self.t_pre_month = StartDate.month  # Record last t month (for RemainMonthlyDiv)
        self.DataLength = DataLength
        self.AgConfig = AgConfig
        
        #--- Load ObvDf from ObvDfPath.
        self.AgObvDf = {}
        self.AgOutput = {}
        self.AgMidResult = {}
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")  
        
        for ag in self.AgList:
            ObvDf = {}
            for k, v in self.AgConfig[ag]["Attributions"]["ObvDfPath"].items():
                if k == "InitDiv":
                    ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0, infer_datetime_format = True)
                else:
                    ObvDf[k] = pd.read_csv(v, index_col=0)
                logger.info("[{}] Load {}: {}".format(ag, k, v))
            self.AgObvDf[ag] = ObvDf
                
            #--- Store space
            Output = pd.DataFrame(index=self.rng, columns=["DailyAction", "RequestDiv", "ActualDiv"])
            self.AgOutput[ag] = Output.to_dict('list')    # Turn df into a dict of list form to avoid df.loc, which is slow.
        
            # Mid-calculation results.
            MidResult = {"RemainMonthlyDiv": 0,
                        "MonthlyDivShortage": []} 
            self.AgMidResult[ag] = MidResult
        
            #--- Check whether to run the DM or assigned values.
            if self.AgConfig[ag]["Inputs"]["DMFreq"] is None:
                self.AssignValue = True
                # One time assignment df.loc is fine.
                self.AgOutput[ag]["DailyAction"][:] = self.AgObvDf[ag]["AssignedBehavior"].loc[self.rng, ag]    
            else:
                self.AssignValue = False
                # Load initial assigned daily diversion.
                self.AgOutput[ag]["DailyAction"][:180] = self.ObvDf["InitDiv"].loc[self.rng[:180]  , ag]   
            
        logger.info("Initialize irrigation diversion agent: {}".format(self.Name))
        
    def act(self, Q, AgentDict, node, CurrentDate, t, DM):
        self.Q = Q
        self.CurrentDate = CurrentDate
        self.t = t
        AgOutput = self.AgOutput
        AgMidResult = self.AgMidResult
        AgList = self.AgList
        
        #==================================================
        #--- Get Output (Diversion decision)
        # For now we hard code the decision period here.
        if self.AssignValue is False: 
            for ag in AgList:
                if self.CurrentDate.month == 3 and self.CurrentDate.day == 1:
                    AgOutput[ag] = DM(Q, t, CurrentDate, ag, AgOutput[ag], AssignValue = False)
        #==================================================
        #==================================================
        #--- Get request list
        # Only deal with diversion not return flow.
        RequestDivList = []
        for ag in AgList:
            MidResult = AgMidResult[ag]
            Output = AgOutput[ag]
            
            if CurrentDate.month != self.t_pre_month:
                MidResult["MonthlyDivShortage"].append(MidResult["RemainMonthlyDiv"])
                MidResult["RemainMonthlyDiv"] = 0
            self.t_pre_month = CurrentDate.month
            RemainMonthlyDiv = MidResult["RemainMonthlyDiv"]
            
            RequestDiv = (Output["DailyAction"][t]) + RemainMonthlyDiv
            Output["RequestDiv"][t] = RequestDiv
            RequestDivList.append(RequestDiv)
            
            AgOutput[ag] = Output
            AgMidResult[ag] = MidResult
        
        #==================================================
        #==================================================
        #--- Proportional discount
        MinFlowTarget = 0   # cms
        TotalRequest = np.sum(RequestDivList)
        AvailableWater = self.Q[node][t] - MinFlowTarget
        if AvailableWater <= 0:
            for i, ag in enumerate(AgList):
                MidResult = AgMidResult[ag]
                Output = AgOutput[ag]
                Output["ActualDiv"][t] = 0
                MidResult["RemainMonthlyDiv"] = RequestDivList[i]
                AgOutput[ag] = Output
                AgMidResult[ag] = MidResult
            Qt = self.Q[node][t]
        elif AvailableWater - TotalRequest >= MinFlowTarget:
            for i, ag in enumerate(AgList):
                MidResult = AgMidResult[ag]
                Output = AgOutput[ag]
                Output["ActualDiv"][t] = RequestDivList[i]
                MidResult["RemainMonthlyDiv"] = 0
                AgOutput[ag] = Output
                AgMidResult[ag] = MidResult
            Qt = self.Q[node][t] - TotalRequest
        else:   # Discount
            Proportion = self.Proportion
            Deficiency = TotalRequest - (AvailableWater - MinFlowTarget)
            # Use the prorated amound of each agent to calculate the proportion.
            for i, ag in enumerate(AgList):
                MidResult = AgMidResult[ag]
                Output = AgOutput[ag]
                Output["ActualDiv"][t] = RequestDivList[i] - Deficiency * Proportion[i]
                MidResult["RemainMonthlyDiv"] = Deficiency * Proportion[i]
                AgOutput[ag] = Output
                AgMidResult[ag] = MidResult
            Qt = MinFlowTarget
        
        self.AgOutput = AgOutput
        self.AgMidResult = AgMidResult
        self.Q[node][t] = Qt
        
        return self.Q
    
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
            
    def act(self, Q, AgentDict, node, CurrentDate, t, DM = None):
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


# b = int(round(Pars["b"], 0))     # Convert to integer
# f = 1 / ( 1 + ( (y - FlowTarget) / L )**(2 * b) )
# if y >= FlowTarget: # increase div decrease c
#     V = - (1 - f)
#     Vavg = np.sum([V] + V_pre)/10     # Average over past ten years
#     c = c + Lr_c * Vavg * c
#     c = c + Lr_c * V * c
# else:
#     V = (1 - f)
#     Vavg = np.sum([V] + V_pre)/10
#     c = c + Lr_c * Vavg * (1-c)
#     c = c + Lr_c * V * (1-c)