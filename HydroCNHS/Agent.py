#%%
import pandas as pd
import numpy as np
from .RL import Value, Policy, Actor_Critic   # RL module

class BasicAgent(object):
    """This is a basic agent class, which defines the connection methods to couple with HydroCNHS and different AgType class.
        Every agent type class has to inherit from BasicAgent to ensure the connectivity to the HydroCNHS.
    """
    def __init__(self, Name, Config, StartDate, DataLength):
        """This is a basic agent class, which defines the connection methods to couple with HydroCNHS.
            Every agent type class has to inherit from BasicAgent to ensure the connectivity to the HydroCNHS.
        Args:
            Name (str): Agent name.
            Config (dict): Model setting dictionary for specific agent.
            StartDate (datetime): Start date of the HydroCNHS simulation.
        """
        self.Name = Name                    # Agent name.   
        self.Qin = None                     # Input outlets' flows.
        self.Qout = None                    # Updated outlets' flows.
        self.DMInfo = None                  # Whatever infomation in a dictionary form could be used by agent to make decision.
        self.StartDate = StartDate          # Datetime object.
        self.CurrentDate = None             # Datetime object.
        self.t = None                       # Current time step index.
        self.t_pre = None                   # Record last t that "act()" is called, which make sure DMFunc() will only be called once in a single time step.
        self.DataLength = DataLength
        self.Inputs = Config["Inputs"]
        self.Attributions = Config.get("Attributions")
        self.Pars = Config["Pars"]
        self.DecisionDF = None              # Results of DMFunc. DF has datetime index and single column.
        self.Active = None                  # Indicator for checking whether its agent's decision-making time step.
        
        # Raise error (Now the StartDate has to be prior to all decision in the first year).
        DMFreq = self.Inputs["DMFreq"]
        if DMFreq.count(None) == 0:
            if DMFreq.count(-9) == 2:
                if StartDate.day > DMFreq[2]:  # every month on day d
                    raise ValueError("The StartDate has to be prior to all decision in the first year.")
            elif DMFreq.count(-9) == 1:
                if StartDate.month > DMFreq[1]:
                    raise ValueError("The StartDate has to be prior to all decision in the first year.")
                elif StartDate.month == DMFreq[1] and StartDate.day > DMFreq[2]:  # every year on m/d
                    raise ValueError("The StartDate has to be prior to all decision in the first year.")
        
        # self.DMFunc = None                  # Assigned decision-making function. (e.g. water demand)
        # self.MinusFunc = None               # Agent's actual action for diverting water.
        # self.PlusFunc = None                # Agent's actual action for releasing water or other adding actions (e.g. return flow).
             
    def checkActiveness(self, CurrentDate):
        """checking whether its agent's decision-making time step.
            Note that DM process only occur at most once in a each time step.

        Args:
            CurrentDate (datetime): Current date.

        Returns:
            bool: Active or not.
        """
        StartDate = self.StartDate
        DMFreq = self.Inputs["DMFreq"] # Coupling frequency setting from DecisionFreq (dict)
        self.Active = False
        # Make sure DMFunc() will only be called once in a single time step.
        if self.t == self.t_pre and self.t_pre is not None:
            return self.Active
        
        if DMFreq.count(None) == 2:   # Type 1 format specify period. e.g. every 2 months.
            if DMFreq[2] is not None:     # day period
                dD = (CurrentDate - StartDate).days
                if dD%DMFreq[2] == 0:
                    self.Active = True
            elif DMFreq[1] is not None:     # month period
                dM = (CurrentDate.year - StartDate.year) * 12 + (CurrentDate.month - StartDate.month)
                if dM%DMFreq[1] == 0 and (CurrentDate.day - StartDate.day) == 0:
                    self.Active = True
            elif DMFreq[0] is not None:     # year period
                dY = CurrentDate.year - StartDate.year
                if dY%DMFreq[0] == 0 and (CurrentDate.month - StartDate.month) == 0 and (CurrentDate.day - StartDate.day) == 0:
                    self.Active = True
        elif DMFreq.count(None) == 0: # Type 2 format specific date. e.g. every year on 1/1
            if DMFreq.count(-9) == 2:
                if CurrentDate.day == DMFreq[2]:  # every month on day d
                    self.Active = True
            elif DMFreq.count(-9) == 1:
                if CurrentDate.month == DMFreq[1] and CurrentDate.day == DMFreq[2]:  # every year on m/d
                    self.Active = True
        self.t_pre = int(self.t)
        return self.Active
    
    def act(self, Q_LSM, AgentDict, node, CurrentDate, t):
        """For agent to update outlet flows.

        Args:
            PlusOrMinus (str): "Plus" or "Minus".
            Q_LSM (array): Input outlets' flows.
            DMInfo (dict): Whatever infomation that will be used by agent to make decision.
            CurrentDate (datetime): Current date.
            t (int): Time step index.

        Returns:
            array: Adjusted outlets' flows for routing.
        """
        self.Qin = Q_LSM
        self.AgentDict = AgentDict
        self.CurrentDate = CurrentDate
        self.t = t
        
        # If agent is active, then use RL to make its decision and store temporal results in self.Decision for PlusFunc and MinusFunc.
        if self.checkActiveness(CurrentDate):
            self.DMFunc()
            
        # Act
        Df = self.DecisionDF
        # Qin = deepcopy(self.Qin)                        # Make sure there are no pointer error.
        # Not necessary deepcopy. We have to update Q_LSM. However, we store the final routing result in Qt.
        # So everything should be fine.
        Qin = self.Qin
        Factor = self.Inputs["Links"][node]
        if isinstance(Factor, list):
            Factor = self.Pars[Factor[0]][Factor[1]]    # e.g. Pars["ReturnFlowFactor"][0]  
        # Hard physical constraint that the streamflow must above or equal to 0.
        # In future, minimum natural flow constraints can be plugged into here.
        # In this case, we didn't use the different between request and actual.
        # Res release constraints are implemented in agent.
        Qin[node][self.t] = max(Qin[node][self.t] + Factor * Df.loc[self.CurrentDate, list(Df)[0]], 0)
        self.Qout = Qin
                
        # return the adjusted outlets' flows for routing.
        return self.Qout
    
    def countNumDM(self, StartDate, EndDate, Freq):
        """Return index with corresponding Freq for self.Record.
            We deal the initial value in agents init when the first DM in a year is before the StartDate.

        Args:
            StartDate (datetime): Simulation start date.
            EndDate (date): Current date.
            Freq (str): D or M or Y

        Returns:
            int: Index for self.Record
        """
        if Freq == "D":
            return (EndDate-StartDate).days + 1
        if Freq == "M":
            return (EndDate.year - StartDate.year) * 12 + (EndDate.month - StartDate.month) + 1
        if Freq == "Y":
            return (EndDate.year - StartDate.year) + 1
            
class AgType_Reservoir(BasicAgent):
    """
    BasicAgent class includes following three main methods:
        __init__(self, Name, Config, StartDate)
        checkActiveness(self, CurrentDate)
        act(self, Q_LSM, AgentDict, node, CurrentDate, t)
        countNumDM(self, StartDate, EndDate, Freq)
    For the AgType class, our main job is to define what agent will act when 
    "act" method is implemented, which is to populate DMFunc class method.
    To fully define act(), the following three methods have to be well-defined.
    Args:
        BasicAgent (Class): See BasicAgent class.
    """
    def __init__(self, Name, Config, StartDate, DataLength):
        super().__init__(Name, Config, StartDate, DataLength)
        # Initialize Actor_Critic for different group!
        ModelAssignList = self.Inputs["ModelAssignList"]      # 0~...
        Pars = self.Pars
        NumModel = len(set(ModelAssignList))               # Total number of different RL models.
        self.Actor_Critic = {}
        for m in range(NumModel):
            ModelPars = {}
            for k, v in Pars.items():
                NumMPars = int(len(v)/NumModel)
                ModelPars[k] = v[NumMPars*m: NumMPars*(m+1)]
            self.Actor_Critic[m] = Actor_Critic(self.Inputs["RL"]["ValueFunc"], 
                                                self.Inputs["RL"]["PolicyFunc"], 
                                                ModelPars, **self.Inputs["RL"]["kwargs"])
        
        # Initialize Record with proper index.
        NumDM = self.countNumDM(StartDate, StartDate + pd.DateOffset(days=self.DataLength), "M")
        # (NumDM+1) since we will include the initial value
        self.Records = {"Actions":              [None]*(NumDM),     # Already include the initial. countNumDM == +1
                        "MonthlyRelease":       [None]*(NumDM),
                        "MonthlyStorage":       [None]*(NumDM),
                        "MonthlyStoragePercent":[None]*(NumDM),
                        "Mu":                   [None]*(NumDM),
                        "Sig":                  [None]*(NumDM)}
        self.Records["MonthlyStorage"][0] = self.Attributions["InitStorage"]
        self.Records["MonthlyStoragePercent"][0] = self.Attributions["InitStorage"]/self.Attributions["Capacity"]*100
        
        # Dynamic release reference 10 yr moving average for each month. 
        self.ResRef = np.zeros((10 ,12))                    # Each column store one month data and each row is one year.
        self.ResRef[:] = np.nan 
        self.ResRef[0,:] = self.Attributions["InitResRef"]  # A list with size 12 (month)
        
        # Will only count it at the first DM and store it for later usage, since the inflow will not be updated in YRB case study.
        self.QSMonthlyDF = None         # Monthly inflow from simulation.
        
        self.actionTuple = None
        
        # Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0)
        
    def getValueFeatureVector(self):
        """Value features = [dQG, dQC]
            dQG     (monthly):  self.ObvDf["MonthlyFlow"].loc[LastMonthDate, "G"] at LastMonthDate (Datetime index).
            dQC     (monthly):  self.ObvDf["MonthlyFlow"].loc[LastMonthDate, "C1"/"C2"] at LastMonthDate (Datetime index).
        Returns:
            array: [dQG, dQC]
        """
        Qin = self.Qin
        LastMonthDate = self.CurrentDate - pd.DateOffset(months=1)
        Scale = self.Attributions["Scale"]                              
        
        # dQC
        if self.Name == "R1":
            if self.countNumDM(self.StartDate, self.CurrentDate, "M") == 1: # Deal with initial time period.
                dQC = 0
            else:
                C_obv = self.ObvDf["MonthlyFlow"].loc[LastMonthDate, "C1"]          # cms (average)
                C = np.mean(Qin["C1"][:self.t-1][-LastMonthDate.daysinmonth:])  # cms (average)
                dQC = (C - C_obv) / Scale["C1"]
        elif self.Name == "R2" or self.Name == "R3":
            
            if self.countNumDM(self.StartDate, self.CurrentDate, "M") == 1: # Deal with initial time period.
                dQC = 0
            else:
                C_obv = self.ObvDf["MonthlyFlow"].loc[LastMonthDate, "C2"]          # cms (average)
                C = np.mean(Qin["C2"][:self.t-1][-LastMonthDate.daysinmonth:]) 
                dQC = (C - C_obv) / Scale["C2"]
        
        # dQG    
        if self.countNumDM(self.StartDate, self.CurrentDate, "M") == 1: # Deal with initial time period.
            dQG = 0
        else:   
            G_obv = self.ObvDf["MonthlyFlow"].loc[LastMonthDate, "G"]               # cms (average)  
            G = np.mean(Qin["G"][:self.t-1][-LastMonthDate.daysinmonth:])           # cms (average) 
            dQG = (G - G_obv) / Scale["G"]
        
        X_value = np.array( [dQG, dQC] ).reshape((-1,1))    # Ensure a proper dimension for RL
        return X_value
    
    def getPolicyFeatureVector(self):
        """Policy features = [dP_forecast, dInflow_forecast, dResS%]
            dP_forecast         (monthly):  self.ObvDf["MonthlydPrep"] at CurrentDate (month).
            dInflow_forecast    (monthly):  self.Qin (daily to monthly) at CurrentDate (month).
            dResS%              (monthly):  self.AgentDict[r].Records["MonthlyStoragePercent"][self.t - 1] (previous day).
                                            ResRef: self.AgentDict[r].ResRef (list of 12 ref ResS% for each month)
        Returns:
            array: [dP_forecast, dInflow_forecast, dResS%]
        """
        CurrentDate = self.CurrentDate.replace(day=1) # convert to the day one of the month
        Scale = self.Attributions["Scale"]                       # Scale 
        
        # dP_forecast (Assume perfect forecast)
        dP_forecast = self.ObvDf["MonthlydPrep"].loc[CurrentDate, self.Name]
        dP_forecast = dP_forecast / Scale["dP"+self.Name]
        
        # dInflow_forecast (Assume perfect forecast from model simulation, not input)
        InflowOutlet = "S" + self.Name[-1]
        if self.QSMonthlyDF is None:    # Then we calculate QSMonthlyDF from Qin from LSM. (One time calculation.)
            Qin = self.Qin
            QS = Qin[InflowOutlet]
            rng = pd.date_range(start = self.StartDate, periods = self.DataLength, freq = "D") 
            df = pd.DataFrame({InflowOutlet: QS}, index=rng)
            self.QSMonthlyDF = df.resample("MS").mean()
            QSMonthlyDF = self.QSMonthlyDF
        else:
            QSMonthlyDF = self.QSMonthlyDF
        QSMonthlyDF = QSMonthlyDF[self.StartDate:CurrentDate]
        QSMonthlyDFm = QSMonthlyDF[QSMonthlyDF.index.month == CurrentDate.month].values.flatten().tolist()
        dInflow_forecast = QSMonthlyDFm[-1] - np.mean(QSMonthlyDFm[-11:])  # Instead of QSMonthlyDFm[:-1][-10:], we take last 11 values including current value to avoid initial value erroe.
        dInflow_forecast = dInflow_forecast / Scale["d"+InflowOutlet]
        
        # dResS% at last month
        LastMonth = CurrentDate.month-1
        if LastMonth == 0: LastMonth = 12
        NumDM = self.countNumDM(self.StartDate, self.CurrentDate, "M")
        
        RList = ["R1", "R2", "R3"] #RList.remove(self.Name)
        dResSPercent = []
        for r in RList:
            # Get MonthlyStoragePercent from last DM from other agents.
            MonthlyStoragePercent = self.AgentDict[r].Records["MonthlyStoragePercent"][NumDM-1]
            ResRef = np.nanmean(self.AgentDict[r].ResRef, axis=0)   # return 1d array with size 12.
            ResRef = ResRef[(LastMonth-1)%12]
            dResSPercent.append( (MonthlyStoragePercent - ResRef) / Scale["dResSPer"+r] )      
        
        X_action = np.array( [dP_forecast, dInflow_forecast] + dResSPercent).reshape((-1,1))    # Ensure a proper dimension for RL
        return X_action
    
    def actionToDailyOutput(self, ReleaseAction):
        # ReleaseAction = Total monthly release (cms).
        # Note ReleaseAction here is not the action directly from actionTuple.
        
        # Take out FixDailyRatio for specific month.
        # Note that Feb = 29 days as default.
        FixDailyRatio = self.ObvDf["FixDailyRatio"][self.Name]
        FixDailyRatio = FixDailyRatio[FixDailyRatio.index.month == self.CurrentDate.month].values.flatten() 
        
        # Non-leap year
        if self.CurrentDate.year%4 != 0 and self.CurrentDate.month == 2:
            FixDailyRatio = FixDailyRatio[:-1]                      # Eliminate 2/29.
        
        # Rescale
        FixDailyRatio = FixDailyRatio/np.sum(FixDailyRatio)     
        
        # Distribute
        Release = ReleaseAction*len(FixDailyRatio)*FixDailyRatio            # cms
        rng = pd.date_range(start = self.CurrentDate, periods = len(FixDailyRatio), freq = "D")
        ReleaseDf = pd.DataFrame({"Release": Release}, index = rng)
        return ReleaseDf
    
    def DMFunc(self):
        """
        Make monthly decision, which return dRelease (action) to update the reference monthly release.
        We define the reference monthly release as the last ten years monthly release average for each month.
        Final results are self.ReleaseDF = ReleaseDF
        """
        # Extract features. (-1, 1)
        X_value = self.getValueFeatureVector()
        X_action = self.getPolicyFeatureVector()
        
        # Actor_Critic make decision.
        ## Select corresponding model
        m = self.Inputs["ModelAssignList"][self.CurrentDate.month-1]
        Actor_Critic = self.Actor_Critic[m]
        Actor_Critic.updatePars(NewX_value = X_value, NewX_action = X_action, R = 0, actionTuple = self.actionTuple)
        actionTuple = Actor_Critic.getAction(X_action)  # (action, mu, sig)
        self.actionTuple = actionTuple
        self.Actor_Critic[m] = Actor_Critic
        
        # Calculate annual divertion and update DivRef (list, keep last ten year).
        ResRef = np.nanmean(self.ResRef,axis=0)   # return 1d array with size 12.
        ResRef = ResRef[(self.CurrentDate.month - 1)%12]
        ReleaseAction = ResRef + actionTuple[0]             # ResRef + dRes
        
        self.ResRef[:, (self.CurrentDate.month - 1)%12] = [ReleaseAction] + list(self.ResRef[:-1, (self.CurrentDate.month - 1)%12])
        
        # Disaggregate and store DivDF to self.DecisionDF for plus and minus.
        ReleaseDf = self.actionToDailyOutput(ReleaseAction)
        self.DecisionDF = ReleaseDf         # DivDf has datetime index single column.    
        
        # Record
        Records = self.Records 
        NumDM = self.countNumDM(self.StartDate, self.CurrentDate, "M")
        Records["Actions"][NumDM] = actionTuple[0]
        Records["Mu"][NumDM] = actionTuple[1]
        Records["Sig"][NumDM] = actionTuple[2]
        Records["MonthlyRelease"][NumDM] = ReleaseAction
        CurrentDate = self.CurrentDate.replace(day=1)     # Convert to the day one of the month (making sure corresponding to the df index)
        
        # Update reservoir storage
        ### cms -> km2-m
        dResS = (self.QSMonthlyDF.loc[CurrentDate, list(self.QSMonthlyDF)[0]] - ReleaseAction)*86400*CurrentDate.days_in_month # cubic meters
        dResS = dResS * 10**(-6)    # m3 to km2-m
        Records["MonthlyStorage"][NumDM] = Records["MonthlyStorage"][NumDM-1] + dResS           
        Records["MonthlyStoragePercent"][NumDM] = Records["MonthlyStorage"][NumDM]/self.Attributions["Capacity"]*100
        self.Records = Records



class AgType_IrrDiversion(BasicAgent):
    """
    BasicAgent class includes following three main methods:
        __init__(self, Name, Config, StartDate)
        checkActiveness(self, CurrentDate)
        act(self, Q_LSM, AgentDict, node, CurrentDate, t)
        countNumDM(self, StartDate, EndDate, Freq)
    For the AgType class, our main job is to define what agent will act when 
    "act" method is implemented, which is to populate DMFunc class method.
    To fully define act(), the following three methods have to be well-defined.
    Args:
        BasicAgent (Class): See BasicAgent class.
    """    
    def __init__(self, Name, Config, StartDate, DataLength):
        super().__init__(Name, Config, StartDate, DataLength)
        # Initialize Actor_Critic!
        self.Actor_Critic = Actor_Critic(self.Inputs["RL"]["ValueFunc"], 
                                         self.Inputs["RL"]["PolicyFunc"], 
                                         self.Pars, **self.Inputs["RL"]["kwargs"])
        
        # Initialize Record with proper index.
        NumDM = self.countNumDM(StartDate, StartDate + pd.DateOffset(days=self.DataLength), "Y")
        # (NumDM+1) since we will include the initial value
        self.Records = {"Actions":              [None]*(NumDM),
                        "AnnualDiv":            [None]*(NumDM),
                        "Mu":                   [None]*(NumDM),
                        "Sig":                  [None]*(NumDM)}
        self.Records["AnnualDiv"][0] = self.Attributions["InitDivRef"]
        
        # Dynamic diversion reference 10 yr moving average. 
        self.DivRef = [self.Attributions["InitDivRef"]]
        # Will only count it at the first DM and store it for later usage, since the inflow will not be updated in YRB case study.
        
        self.actionTuple = None
        
        # Load ObvDf from ObvDfPath.
        self.ObvDf = {}
        for k, v in self.Attributions["ObvDfPath"].items():
            self.ObvDf[k] = pd.read_csv(v, parse_dates=True, index_col=0)
            
        # Assign initial DM (before first 3/1)
        DivDf = self.actionToDailyOutput(DivAction = 1, Initial = True)
        self.DecisionDF = DivDf         # DivDf has datetime index single column.    
        
    def getValueFeatureVector(self):
        """Value features = [dQG]
            dQG     (annually):  self.ObvDf["AnnualFlow"].loc[LastYearDate, "G"] (Datetime index).
        Returns:
            array: [dQG]
        """
        Scale = self.Attributions["Scale"]   
        Qin = self.Qin
        LastYearDate = (self.CurrentDate - pd.DateOffset(years=1)).replace(month=3).replace(day=1)

        if self.countNumDM(self.StartDate, self.CurrentDate, "Y") == 1: # Assume the first DM has no 0 value feature.
            dQG = 0
        else:    
            G_obv = self.ObvDf["AnnualFlow"].loc[LastYearDate, "G"]     # cms (average)
            G = np.mean(Qin["G"][:self.t-1][-365:])                     # cms (average) Ignore leap year thing.
            dQG = (G - G_obv) / Scale["G"]
        
        X_value = np.array( [dQG] ).reshape((-1,1))    # Ensure a proper dimension for RL
        return X_value
    
    def getPolicyFeatureVector(self):
        """Policy features = [dP_forecast, dResS%]
            dP_ResWTotal        (monthly):  self.ObvDf["AnnualdPrep"] at CurrentDate (month).
            dResS%              (monthly):  self.AgentDict[r].Records["MonthlyStoragePercent"][self.t - 1] (previous day).
                                            ResRef: self.AgentDict[r].ResRef (list of 12 ref ResS% for each month)
        Returns:
            array: [dP_ResWTotal, dResS%]
        """
        CurrentDate = self.CurrentDate.replace(month=3).replace(day=1) # convert to the day one of the month
        Scale = self.Attributions["Scale"]                                       # Scale 
        
        # dP_ResWTotal (observation)
        dP_ResWTotal = self.ObvDf["AnnualdPrep"].loc[CurrentDate, "dPResWTotal"]
        dP_ResWTotal = dP_ResWTotal / Scale["dPResWTotal"]
        
        
        # dResS% at last month
        NumDM = self.countNumDM(self.StartDate, self.CurrentDate, "M") - 1 # Last month's storage.
        
        RList = ["R1", "R2", "R3"] #RList.remove(self.Name)
        dResSPercent = []
        for r in RList:
            # Get MonthlyStoragePercent from current DM from reservoir agents.
            MonthlyStoragePercent = self.AgentDict[r].Records["MonthlyStoragePercent"][NumDM]
            ResRef = np.nanmean(self.AgentDict[r].ResRef, axis=0)   # return 1d array with size 12.
            ResRef = ResRef[(self.CurrentDate.month-1)%12]
            dResSPercent.append( (MonthlyStoragePercent - ResRef) / Scale["dResSPer"+r] )      
        
        X_action = np.array( [dP_ResWTotal] + dResSPercent).reshape((-1,1))    # Ensure a proper dimension for RL
        return X_action
    
    def actionToDailyOutput(self, DivAction, Initial = False):
        # DivAction = Total annual diversion (cms).
        # Note DivAction here is not the action directly from actionTuple.
        
        # Since the decision is made on 3/1, we idenfy leap year for current year + 1.
        # Note that the FixDailyRatio should start at 3/1
        FixDailyRatio = self.ObvDf["FixDailyRatio"]
        if FixDailyRatio.index[0].year != 2000:
            FixDailyRatio.index = [date.replace(year=2000) for date in FixDailyRatio.index]
        
        CurrentDate = self.CurrentDate
        if Initial is False:
            if (self.CurrentDate.year + 1)%4 != 0:
                FixDailyRatio = pd.concat([FixDailyRatio["2000-3-1":], FixDailyRatio[:"2000-2-28"]])
            else:
                FixDailyRatio = pd.concat([FixDailyRatio["2000-3-1":], FixDailyRatio[:"2000-2-29"]])
            FixDailyRatio = FixDailyRatio[self.Name].values.flatten() 
            # Rescale
            FixDailyRatio = FixDailyRatio/np.sum(FixDailyRatio)     
            # Distribute
            Div = DivAction*len(FixDailyRatio)*FixDailyRatio            # cms
        else:       # For temp initial DM
            Div = FixDailyRatio[self.Name].values.flatten() 
            CurrentDate = self.StartDate
            
        rng = pd.date_range(start = CurrentDate, periods = len(FixDailyRatio), freq = "D")
        DivDf = pd.DataFrame({"Div": Div}, index = rng)
        return DivDf
    
    def DMFunc(self):
        """
        Make annual decision, which return dDiv (action) to update the reference annual diversion.
        We define the reference annual diversion as the last ten years diversion average.
        Final results are self.DecisionDF = DivDf
        """
        # Extract features. (-1, 1)
        X_value = self.getValueFeatureVector()
        X_action = self.getPolicyFeatureVector()
        
        # Actor_Critic make decision.
        Actor_Critic = self.Actor_Critic
        Actor_Critic.updatePars(NewX_value = X_value, NewX_action = X_action, R = 0, actionTuple = self.actionTuple)
        actionTuple = Actor_Critic.getAction(X_action)  # (action, mu, sig)
        self.Actor_Critic = Actor_Critic
        self.actionTuple = actionTuple
        
        # Calculate annual divertion and update DivRef (list, keep last ten year).
        DivRef = np.mean(self.DivRef)
        DivAction = DivRef + actionTuple[0]             # DivRef + dDiv
        self.DivRef = self.DivRef[-9:] + [DivAction]
        
        # Disaggregate and store DivDF to self.DecisionDF.
        DivDf = self.actionToDailyOutput(DivAction)
        self.DecisionDF = DivDf         # DivDf has datetime index single column.    
        
        # Record
        Records = self.Records 
        NumDM = self.countNumDM(self.StartDate, self.CurrentDate, "Y")
        Records["Actions"][NumDM] = actionTuple[0]
        Records["Mu"][NumDM] = actionTuple[1]
        Records["Sig"][NumDM] = actionTuple[2]
        Records["AnnualDiv"][NumDM] = DivAction
        self.Records = Records



#%%
# Ag1 = {
#     "Inputs":{
#         "Piority": 1,
#         "DMFreq": [-9, 1, 1]},
#     "Links":{  
#         "S1": -1,                                  # Add to Q of outlet "A" by -1*Action
#         "S2": 0.2},                                # Add to Q of outlet "B" by 0.2*Action
#     "Pars":{                                    
#         "LearnRate":  0.1,                         # RL
#         "W":      [0.5,0.5],                       # RL
#         "Theta":  [0.5,0.5]}}
# ag = AgType_IrrDiversion("Ag1", Ag1, "1010")
# ag.act(PlusOrMinus, Q_LSM, DMInfo, CurrentDate, t)
# %%
# def MinusFunc(self, node):
#     """This function will be run at each t if the agent was called.
#     """
#     print("Here is MinusFunc")
#     DivDf = self.DecisionDF
#     Qin = deepcopy(self.Qin)        # Make sure there are no pointer error.
#     Factor = self.Inputs["Links"][node]
#     Qin[node][self.t] = Qin[node][self.t] + Factor * DivDf.loc[self.CurrentDate, list(DivDf)[0]]
#     self.Qout = Qin
    
# def PlusFunc(self, node):
#     """This function will be run at each t if the agent was called.
#     """
#     print("Here is PlusFunc")
#     DivDf = self.DecisionDF
#     Qin = deepcopy(self.Qin)        # Make sure there are no pointer error.
#     Factor = self.Inputs["Links"][node]
#     if isinstance(Factor, list):
#         Factor = self.Pars[Factor[0]][Factor[1]]    # e.g. Pars["ReturnFlowFactor"][0]  
#     Qin[node][self.t] = Qin[node][self.t] + Factor * DivDf.loc[self.CurrentDate, list(DivDf)[0]]
#     self.Qout = Qin