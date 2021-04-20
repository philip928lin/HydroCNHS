import pandas as pd

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
        if self.AssignValue:
            # Expect to be a df.
            self.AssignedBehavior = self.ObvDf["AssignedBehavior"]    
            
        # Storage
        self.TempResult = {"RemainMonthlyDiv": 0}
        self.Result = {"MonthlyDivShortage": [],
                       "ActualDiv": [],
                       "RequestDiv": []}  
    
    def act(self, Q, AgentDict, node, CurrentDate, t):
        self.Q = Q
        self.AgentDict = AgentDict
        self.CurrentDate = CurrentDate
        self.t = t
        
        if CurrentDate.month != self.t_pre_month:
            self.Result["MonthlyDivShortage"].append(self.TempResult["RemainMonthlyDiv"])
            self.TempResult["RemainMonthlyDiv"] = 0
        RemainMonthlyDiv = self.TempResult["RemainMonthlyDiv"]
        
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]
        
        # Diversion
        if Factor < 0:
            RequestDiv = self.AssignedBehavior.loc[CurrentDate, self.Name] + RemainMonthlyDiv
            Qorg = self.Q[node][self.t]
            MinFlowTarget = 0   # cms
            
            if Qorg <= MinFlowTarget:
                ActualDiv = 0
                RemainMonthlyDiv = RequestDiv
                Qt = Qorg
            else:
                Qt = max(Qorg + Factor * RequestDiv, MinFlowTarget)
                ActualDiv = Qorg - Qt   # >= 0
                RemainMonthlyDiv = max(RequestDiv-ActualDiv, 0)
            
            self.Result["RequestDiv"].append(RequestDiv)
            self.Result["ActualDiv"].append(ActualDiv)
            self.TempResult["RemainMonthlyDiv"] = RemainMonthlyDiv
            self.Q[node][self.t] = Qt
            return Q
        
        elif Factor > 0:
            # Assume that diversion has beed done in t.
            Div_t = self.Result["ActualDiv"][-1]
            self.Q[node][self.t] = self.Q[node][self.t] + Factor * Div_t
            
            
        
        
            