import os
import pandas as pd
import numpy as np
from scipy.stats import norm, truncnorm, multivariate_normal, uniform, beta
from dateutil.relativedelta import relativedelta
import logging

logger = logging.getLogger("ABM") 

class DivDM(object):
    def __init__(self, StartDate, DataLength, ABM):
        BasicPath = r"C:\Users\ResearchPC\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\YRBModel\INPUT"
        
        # These are default path
        self.Path = {"FlowTarget": os.path.join(BasicPath, "FlowTarget_Cali.csv"),
                     "Database": os.path.join(BasicPath, "Database_1959_2013.csv"),
                     "CCurves": r"C:\Users\ResearchPC\OneDrive\Lehigh\0_Proj2_UA-SA-Equifinality\Data",
                     "AgCor": os.path.join(BasicPath, "AgCor.csv"),
                     "InitDiv": os.path.join(BasicPath, "Diversion_D_cms.csv") }
        # Input prec scenario
        DatabaseScenarioPath = ABM["Inputs"].get("Database")
        if DatabaseScenarioPath is not None:
            self.Path["Database"] = DatabaseScenarioPath
        
        # FlowTarget
        FlowTargetPath = ABM["Inputs"].get("FlowTarget")
        if DatabaseScenarioPath is not None:
            self.Path["FlowTarget"] = FlowTargetPath
        
        self.AgList = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
        self.Obv = {}
        self.Ag = {}
        self.t = None
        self.t_pre = None
        self.StartDate = StartDate
        self.DataLength = DataLength
        self.SocialNorm = False
        self.BehaviorType = ABM["Inputs"].get("BehaviorType")
        if self.BehaviorType is None:           # Learning/Adaptive/Static
            self.BehaviorType = "Learning"      # Default
            logger.warn("Agent's BehaviorType is set to 'Learning' as default.")
        self.DMcount = 0
        
        #--- AgPars, AgInputs (Not optimize)
        self.AgPars = {}
        self.AgInputs = {}
        for ag in self.AgList:
            for agType in ABM:
                if ag in ABM[agType]:
                    self.AgPars[ag] = ABM[agType][ag]["Pars"]
                    self.AgInputs[ag] = ABM[agType][ag]["Inputs"]
        
        #--- InitDiv
        InitDivPath = ABM["Inputs"].get("InitDiv")
        if InitDivPath is not None:
            self.Path["InitDiv"] = InitDivPath
        # Only consider when start year is nonleap year
        InitDiv = pd.read_csv(self.Path["InitDiv"], parse_dates=True, index_col=0, \
                              infer_datetime_format = True)["{}-1-1".format(StartDate.year):"{}-2-28".format(StartDate.year)]
        
        #--- Store space
        rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")
        self.rng = rng
        DMNum = len(pd.date_range(start = rng[0], end = rng[-1], freq = "Y"))
        for ag in self.AgList:
            self.Ag[ag] = {"RL":{"y": [None]*DMNum,
                                 "x": [None]*DMNum,
                                 "V": [None]*DMNum,
                                 "Vavg": [None]*DMNum,
                                 "Mu": [None]*DMNum,
                                 "DivReqRef": [None]*DMNum,
                                 "YDivReq": [None]*DMNum },
                           "Init": {"DivReqRef": self.AgInputs[ag]["InitYDivRef"]},
                           "DailyAction": list(InitDiv[ag])}
        self.IncreaseMinFlow = 0
        
        #--- Flowtarget
        FlowTarget = pd.read_csv(self.Path["FlowTarget"], index_col=0)
        self.FlowTarget = FlowTarget.to_dict('dict')["FlowTarget"]
        
        #--- Feature - TotalDamS
        # TotalDamS = pd.read_csv(self.Path["TotalDamS"], index_col=0)
        # self.TotalDamS = {"Index": TotalDamS.index,
        #                   "Value": TotalDamS.to_numpy().flatten()}
        Database = pd.read_csv(self.Path["Database"], index_col=0)
        self.Database = {"Index": Database.index,
                         "Column": Database.columns,
                         "Value": Database.to_numpy()}
  
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
        
        # Only run DMFunc when it has not been done at t. DMFunc will go through all agents.
        if t != self.t_pre:
            self.DMFunc()
            self.t_pre = t
        #Ag = self.Ag
        Output["DailyAction"][:len(self.Ag[ag]["DailyAction"])] = self.Ag[ag]["DailyAction"]
        return Output
    
    def DMFunc(self):
        AgList = self.AgList
        CurrentDate = self.CurrentDate
        FlowTarget = self.FlowTarget[CurrentDate.year - 1]
        DataLength = self.DataLength
        t = self.t
        DMcount = self.DMcount
        
        #==================================================
        # Learning: Totally empirical
        #==================================================
        if self.BehaviorType == "Learning":
            if CurrentDate.year == self.StartDate.year:
                y = FlowTarget   # Initial value (No deviation from the flow target.)
            else:
                mask = [True if i.month in [7,8,9] and (i.year == CurrentDate.year - 1) else False for i in self.rng]
                y = np.mean(self.Q["G"][mask])
            
            IncreaseMinFlow = []
            for ag in AgList:
                Init = self.Ag[ag]["Init"]
                RL = self.Ag[ag]["RL"]
                Pars = self.AgPars[ag]
                MaxYDiv = self.AgInputs[ag]["MaxYDiv"]
                MinYDiv = self.AgInputs[ag]["MinYDiv"]
                
                if DMcount == 0:
                    DivReqRef = Init["DivReqRef"]
                else:
                    DivReqRef = RL["DivReqRef"][DMcount-1]
                L_U = Pars["L_U"]
                L_L = Pars["L_L"]
                Lr_c = Pars["Lr_c"]
                if y > FlowTarget + L_U:
                    V = 1
                elif y < FlowTarget - L_L:
                    V = -1
                else:
                    V = 0
                RL["V"][DMcount] = V
                # Mean value of the past "ten" years.
                Vavg = np.sum(RL["V"][max(0,DMcount-9):DMcount+1])/10   # First few years, strengh is decreased on purpose.  
                DivReqRef = DivReqRef + Vavg*Lr_c*5     # Scale to 5 Lr_c in [0,1]
                if MaxYDiv is not None and MinYDiv is not None:
                    DivReqRef = min(MaxYDiv, max(DivReqRef, MinYDiv))       # Bound by Max and Min
                # Save
                RL["y"][DMcount] = y
                RL["DivReqRef"][DMcount] = DivReqRef
                RL["Vavg"][DMcount] = Vavg
                IncreaseMinFlow.append(DivReqRef - self.AgInputs[ag]["InitYDivRef"])
                self.Ag[ag]["RL"] = RL
            self.IncreaseMinFlow = sum(IncreaseMinFlow)
        else:
            for ag in AgList:
                self.Ag[ag]["RL"]["DivReqRef"][DMcount] = self.AgInputs[ag]["InitYDivRef"]
            
        #==================================================


        #==================================================
        # Adaptive & Emergency Operation (Drought year proration) 
        #==================================================
        #--- Get feature: Annual daily mean BCPrec from 11 - 6
        Database = self.Database
        Index = np.where(Database["Index"] == CurrentDate.year)[0][0] 
        x = Database['Value'][Index, 0]
        
        #--- Get Multinormal random noise
        if self.SocialNorm:
            Corr = self.Corr
            rn = multivariate_normal.rvs(cov = Corr, size=1)
        else:
            rn = norm.rvs(size=5)
        
        #--- Get YDivReq
        for i, ag in enumerate(AgList):
            Pars = self.AgPars[ag]
            RL = self.Ag[ag]["RL"]
            MaxYDiv = self.AgInputs[ag]["MaxYDiv"]
            MinYDiv = self.AgInputs[ag]["MinYDiv"]
            DivReqRef = RL["DivReqRef"][DMcount]
            ProratedRatio = Pars["ProratedRatio"]
            
            if self.BehaviorType == "Static":
                # No stochastic, No adaptive, A constant.
                b = Pars["b"]
                YDivReq = DivReqRef + b
                Mu = YDivReq
            else:
                a = Pars["a"]
                b = Pars["b"]
                sig = Pars["Sig"]
                if x <= 0.583:      # Emergency Operation (Drought year proration) 
                    Mu = DivReqRef * ProratedRatio
                else:               # Adaptive behavoir under normal year.
                    Mu = DivReqRef + a*x+b
                YDivReq = Mu + rn[i]*sig
                
            # Hard constraint for MaxYDiv and MinYDiv
            if MaxYDiv is not None and MinYDiv is not None:
                YDivReq = min(MaxYDiv, max(YDivReq, MinYDiv))       # Bound by Max and Min

            #--- Save
            RL["x"][DMcount] = x
            RL["Mu"][DMcount] = Mu
            RL["YDivReq"][DMcount] = YDivReq
            self.Ag[ag]["RL"] = RL
        #==================================================
        
        #==================================================
        # To Daily
        #==================================================
        for ag in AgList:
            RL = self.Ag[ag]["RL"]
            DailyAction = self.Ag[ag]["DailyAction"]
            CCurves = self.CCurves[ag]
            YDivReq = RL["YDivReq"][DMcount]

            #--- Map back to daily diversion (from Mar to Feb)
            def getMonthlyDiv(YDivReq, a, b, LB, UB):
                if YDivReq <= LB:
                    MDivReq = b
                elif YDivReq >= UB:
                    MDivReq = a*(UB-LB) + b
                else:
                    MDivReq = a*(YDivReq-LB) + b
                return MDivReq
            MRatio = np.array([getMonthlyDiv(YDivReq, *CCurves[m-1]) for m in [3,4,5,6,7,8,9,10,11,12,1,2]])
            MRatio = MRatio/sum(MRatio)
            MDivReq = YDivReq * 12 * MRatio

            #--- To daily. Uniformly assign those monthly average diversion to each day.
            if (CurrentDate + relativedelta(years=1)).is_leap_year :
                DayInMonth = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29]   # From Mar to Feb
                NumDay = 366
            else:
                DayInMonth = [31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28]   # From Mar to Feb
                NumDay = 365
                
            DDiv = []
            for m in range(12):
                DDiv += [MDivReq[m]]*DayInMonth[m]
                
            #--- Store into dataframe. 
            if DataLength - t > 366:
                DailyAction += DDiv
                self.length = NumDay
            else:     # For last year~~~
                DailyAction += DDiv[:self.DataLength - t]
                self.length = self.DataLength - t
            
            #--- Save
            self.Ag[ag]["DailyAction"] = DailyAction
        #==================================================
        self.DMcount += 1
        
        
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
            if k == "InitDiv" or k == "AssignedBehavior":
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
            if CurrentDate.month == 3 and CurrentDate.day == 1:
                Output = DM(Q, t, CurrentDate, ag, Output, AssignValue = False)
        #==================================================
        #==================================================
        #--- Calculate the actual request.
        Factor = self.Inputs["Links"][node]
        # For parameterized (for calibration) factor.
        if isinstance(Factor, list):    
            Factor = self.Pars[Factor[0]][Factor[1]]      

        # Diversion
        if Factor < 0:
            if CurrentDate.month != self.t_pre_month or CurrentDate == self.rng[-1]:
                MidResult["MonthlyDivShortage"].append(MidResult["RemainMonthlyDiv"])
                MidResult["RemainMonthlyDiv"] = 0
            
            self.t_pre_month = CurrentDate.month
            RemainMonthlyDiv = self.MidResult["RemainMonthlyDiv"]
            
            RequestDiv = (-Factor * Output["DailyAction"][t]) + RemainMonthlyDiv
            MinFlowTarget = 3.53 + DM.IncreaseMinFlow  # cms
            AvailableWater = self.Q[node][t] - MinFlowTarget
            
            if AvailableWater <= 0:
                ActualDiv = 0
                RemainMonthlyDiv = RequestDiv
                Qt = self.Q[node][t]
            elif AvailableWater - RequestDiv >= MinFlowTarget:
                ActualDiv = RequestDiv
                RemainMonthlyDiv = 0
                Qt = self.Q[node][t] - RequestDiv
            else:   # Discount
                Deficiency = RequestDiv - (AvailableWater - MinFlowTarget)
                ActualDiv = RequestDiv - Deficiency
                RemainMonthlyDiv = Deficiency
                Qt = MinFlowTarget
            
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
    def __init__(self, Name, Config, StartDate, DataLength):
        self.Name = Name                    # Agent name.   
        self.AgList = ['Roza', 'Wapato', 'Sunnyside']
        self.Proportion = [1, 0.533851525, 0.352633532] # Poratable [0.44, 0.39, 0.17]
        self.StartDate = StartDate          # Datetime object.
        self.t_pre_month = StartDate.month  # Record last t month (for RemainMonthlyDiv)
        self.DataLength = DataLength
        self.AgConfig = Config
        
        #--- Load ObvDf from ObvDfPath.
        self.AgObvDf = {}
        self.AgOutput = {}
        self.AgMidResult = {}
        self.rng = pd.date_range(start = StartDate, periods = DataLength, freq = "D")  
        
        for ag in self.AgList:
            ObvDf = {}
            for k, v in self.AgConfig[ag]["Attributions"]["ObvDfPath"].items():
                if k == "InitDiv" or k == "AssignedBehavior":
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
                self.AgOutput[ag]["DailyAction"][:180] = self.AgObvDf[ag]["InitDiv"].loc[self.rng[:180], ag]   
            
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
            
            if CurrentDate.month != self.t_pre_month or CurrentDate == self.rng[-1]:
                MidResult["MonthlyDivShortage"].append(MidResult["RemainMonthlyDiv"])
                MidResult["RemainMonthlyDiv"] = 0
            RemainMonthlyDiv = MidResult["RemainMonthlyDiv"]
            RequestDiv = (Output["DailyAction"][t]) + RemainMonthlyDiv
            Output["RequestDiv"][t] = RequestDiv
            RequestDivList.append(RequestDiv)
            
            AgOutput[ag] = Output
            AgMidResult[ag] = MidResult
        self.t_pre_month = CurrentDate.month
        #==================================================
        #==================================================
        #--- Proportional discount
        MinFlowTarget = 3.53 + DM.IncreaseMinFlow # cms
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
        elif AvailableWater - TotalRequest >= MinFlowTarget or TotalRequest == 0:
            for i, ag in enumerate(AgList):
                MidResult = AgMidResult[ag]
                Output = AgOutput[ag]
                Output["ActualDiv"][t] = RequestDivList[i]
                MidResult["RemainMonthlyDiv"] = 0
                AgOutput[ag] = Output
                AgMidResult[ag] = MidResult
            Qt = self.Q[node][t] - TotalRequest
        else:   # Discount
            Deficiency = TotalRequest - (AvailableWater - MinFlowTarget)
            Proportion = self.Proportion
            # Solve discount ratio
            r = Deficiency/(sum( [p*req for p, req in zip(Proportion, RequestDivList)] ))
            
            if r > 1:       # If Roza is not enoungh
                r = 1
                Def = [r*p*req for p, req in zip(Proportion, RequestDivList)]
                Remain = (Deficiency-sum(Def)) * np.array(Proportion[1:]) # No Roza
                Def[1] = Def[1] + Remain[0]
                Def[2] = Def[2] + Remain[1]
                if Def[1] > RequestDivList[1]:  # If Wapato is not enough.
                    Remain2 = Def[1] - RequestDivList[1]
                    Def[1] = RequestDivList[1]
                    Def[2] = Def[2] + Remain2
                Deficiency = Def
            else:
                Deficiency = [r*p*req for p, req in zip(Proportion, RequestDivList)]
            
            # Use the prorated amound of each agent to calculate the proportion.
            for i, ag in enumerate(AgList):
                MidResult = AgMidResult[ag]
                Output = AgOutput[ag]
                Output["ActualDiv"][t] = RequestDivList[i] - Deficiency[i]
                MidResult["RemainMonthlyDiv"] = Deficiency[i]
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