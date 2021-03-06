#%%
# Form the water system using a semi distribution hydrological model and agent-based model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

from .LSM import runGWLF, calPEt_Hamon
from .Routing import formUH_Lohmann, runTimeStep_Lohmann
from .SystemConrol import loadConfig, loadModel
from joblib import Parallel, delayed    # For parallelization
from pandas import date_range, to_datetime
from tqdm import tqdm
from copy import deepcopy   # For deepcopy dictionary.
import numpy as np
import time
import logging
logger = logging.getLogger("HydroCNHS") # Get logger 

class HydroCNHS(object):
    """Main HydroCNHS simulation object.

    """
    def __init__(self, model, name = None):
        """HydroCNHS constructor

        Args:
            model (str/dict): model.yaml file (prefer) or dictionary. 
            name ([str], optional): Object name. Defaults to None.
        """
        self.__name__ = name
        
        # Load HydroCNHS system configuration.
        self.Config = loadConfig()   
            
        # Load model.yaml and distribute into several variables.
        Model = loadModel(model)    # We design model to be str or dictionary.
        
        # Need to verify Model contain all following keys.
        
        try:                   
            self.Path = Model["Path"]
            self.WS = Model["WaterSystem"]     # WS: Water system
            self.LSM = Model["LSM"]            # LSM: Land surface model
            self.RR = Model["Routing"]         # RR: Routing 
            self.ABM = Model["ABM"] 
            self.SysPD = Model["SystemParsedData"]
        except:
            logger.error("Model is incomplete for CNHS.")
        # Initialize output
        self.Q = {}     # [cms] Streamflow for routing outlets (Gauged outlets and inflow outlets of in-stream agents).
        RoutingOutlets = self.SysPD["RoutingOutlets"]
        for ro in RoutingOutlets:
            self.Q[ro] = np.zeros(self.WS["DataLength"])
             
        self.A = {}     # Collect agent's output for each AgentType.
    
    def loadWeatherData(self, T, P, PE = None, LSMOutlets = None):
        """[Include in run] Load temperature and precipitation data.
        Can add some check functions or deal with miss values here.
        Args:
            T (dict): [degC] Daily mean temperature time series data (value) for each sub-basin named by its outlet.
            P (dict): [cm] Daily precipitation time series data (value) for each sub-basin named by its outlet.
            PE(dict/None): [cm] Daily potential evapotranpiration time series data (value) for each sub-basin named by its outlet.
            LSMOutlets(dict/None): Should equal to self.WS["Outlets"]
        """
        if PE is None:
            PE = {}
            # Default to calculate PE with Hamon's method and no dz adjustment.
            for sb in LSMOutlets:
                PE[sb] = calPEt_Hamon(T[sb], self.LSM[sb]["Inputs"]["Latitude"], self.WS["StartDate"], dz = None)
        self.Weather = {"T":T, "P":P, "PE":PE}
        logger.info("Load T & P & PE with total length {}.".format(self.WS["DataLength"]))
    
    def checkActiveAgTypes(self, StartDate, CurrentDate):
        """Check the decision point accross agent types.

        Args:
            StartDate (datetime): Start date.
            CurrentDate (datetime): Current date.

        Returns:
            list: List of agent types that make decisions.
        """
        DecisionFreq = self.ABM["DecisionFreq"] # Coupling frequency setting from DecisionFreq (dict)
        ActiveAgTypes = []
        for agType in DecisionFreq:
            DeciFreq = DecisionFreq[agType]
            if DeciFreq.count(None) == 2:   # Type 1 format specify period. e.g. every 2 months.
                if DeciFreq[2] is not None:     # day period
                    dD = (CurrentDate - StartDate).days
                    if dD%DeciFreq[2] == 0:
                        ActiveAgTypes.append(agType)
                elif DeciFreq[1] is not None:     # month period
                    dM = (CurrentDate.year - StartDate.year) * 12 + (CurrentDate.month - StartDate.month)
                    if dM%DeciFreq[1] == 0 and (CurrentDate.day - StartDate.day) == 0:
                        ActiveAgTypes.append(agType)
                elif DeciFreq[0] is not None:     # year period
                    dY = CurrentDate.year - StartDate.year
                    if dY%DeciFreq[0] == 0 and (CurrentDate.month - StartDate.month) == 0 and (CurrentDate.day - StartDate.day) == 0:
                        ActiveAgTypes.append(agType)
            elif DeciFreq.count(None) == 0: # Type 2 format specific date. e.g. every year on 1/1
                if DeciFreq.count(-9) == 2:
                    if CurrentDate.day == DeciFreq[2]:  # every month on day d
                        ActiveAgTypes.append(agType)
                elif DeciFreq.count(-9) == 1:
                    if CurrentDate.month == DeciFreq[1] and CurrentDate.day == DeciFreq[2]:  # every year on m/d
                        ActiveAgTypes.append(agType)
        return ActiveAgTypes    
        
    def __call__(self, T, P, PE = None, AssignedQ = {}, AssignedUH = {}):
        """Run HydroCNHS simulation. The simulation is controled by model.yaml and Config.yaml (HydroCNHS system file).
        
        Args:
            T (dict): Daily mean temperature.
            P (dict): Daily precipitation.
            PE (dict, optional): Potential evapotranspiration. Defaults to None (calculted by Hamon's method).
            AssignedQ (dict, optional): If user want to manually assign Q (value, Array) for certain outlet (key, str). Defaults to None.
            AssignedUH (dict, optional): If user want to manually assign UH (Lohmann) (value, Array) for certain outlet (key, str). Defaults to None.
        """
        
        # Set a timer here
        start_time = time.monotonic()
        self.elapsed_time = 0
        def getElapsedTime():
            elapsed_time = time.monotonic() - start_time
            self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            return self.elapsed_time
        
        Paral = self.Config["Parallelization"]
        Outlets = self.WS["Outlets"]
        self.Q_LSM = {}  # Temporily store Q result from land surface simulation.
        
        # ----- Land surface simulation ---------------------------------------
        if self.LSM["Model"] == "GWLF":
            # Remove sub-basin that don't need to be simulated. Not preserving element order in the list.
            Outlets_GWLF = list(set(Outlets) - set(AssignedQ.keys()))  
            if AssignedQ != {}:
                RoutingOutlets = self.SysPD["RoutingOutlets"]
                for ro in RoutingOutlets:
                    for sb in self.RR[ro]:
                        if sb in AssignedQ:
                            self.SysPD["RoutingOutlets"][ro][sb]["Pars"]["GShape"] = None   # No in-grid routing.
                            self.SysPD["RoutingOutlets"][ro][sb]["Pars"]["GScale"] = None   # No in-grid routing.
                            logger.info("Turn {}'s GShape and GScale in routing setting to None. Since Q (assuming to be observed data) is given, there is no in-grid time lag.".format((sb, ro)))
            logger.info("[{}] Start GWLF for {} sub-basins. [{}]".format(self.__name__, len(Outlets_GWLF), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets_GWLF)    
            # Start GWLF simulation in parallel.
            QParel = Parallel(n_jobs = Paral["Cores_runGWLF"], verbose = Paral["verbose"]) \
                            ( delayed(runGWLF)\
                              (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.Weather["PE"][sb], self.WS["StartDate"], self.WS["DataLength"]) \
                              for sb in Outlets_GWLF ) 
            # Add user assigned Q first.
            self.Q_LSM = deepcopy(AssignedQ)      
            # Collect QParel results
            for i, sb in enumerate(Outlets_GWLF):
                self.Q_LSM[sb] = QParel[i]
            logger.info("[{}] Complete GWLF... [{}]".format(self.__name__, getElapsedTime()))
        # ---------------------------------------------------------------------    
    

        # ----- Form UH for Lohmann routing method ----------------------------
        if self.RR["Model"] == "Lohmann":
            self.UH_Lohmann = {}    # Initialized the output dictionary
            RoutingOutlets = self.SysPD["RoutingOutlets"]
            # Form combination
            UH_List = [(sb, ro) for ro in RoutingOutlets for sb in self.RR[ro]]
            # Remove assigned UH from the list. Not preserving element order in the list.
            UH_List_Lohmann = list(set(UH_List) - set(AssignedUH.keys()))
            # Start forming UH_Lohmann in parallel.
            logger.info("[{}] Start forming {} UHs for Lohmann routing. [{}]".format(self.__name__, len(UH_List_Lohmann), getElapsedTime()))
            UHParel = Parallel(n_jobs = Paral["Cores_formUH_Lohmann"], verbose = Paral["verbose"]) \
                            ( delayed(formUH_Lohmann)\
                            (self.RR[pair[1]][pair[0]]["Inputs"], self.RR[pair[1]][pair[0]]["Pars"]) \
                            for pair in UH_List_Lohmann )     # pair = (Outlet, GaugedOutlet)
            # Add user assigned UH first.

            # Form UH
            self.UH_Lohmann = deepcopy(AssignedUH)
            for i, pair in enumerate(UH_List_Lohmann):
                self.UH_Lohmann[pair] = UHParel[i]
            logger.info("[{}] Complete forming UHs for Lohmann routing... [{}]".format(self.__name__, getElapsedTime()))
        # ---------------------------------------------------------------------

        
        # ----- Time step simulation (Coupling hydrological model and ABM)-----
        # Note: Only Qt of GaugedOutlet1 and the 
        # Obtain datetime index
        StartDate = to_datetime(self.WS["StartDate"], format="%Y/%m/%d")        
        pdDatedateIndex = date_range(start = StartDate, periods = self.WS["DataLength"], freq = "D")    
        SimSeq = self.SysPD["SimSeq"]
        AgSimSeq = self.SysPD["AgSimSeq"]
        InStreamAgents = self.SysPD["InStreamAgents"]
        
        Agents = {}     # Here we store all agent objects with key = agent name.
        
        # Add InStreamAgents to self.Q_LSM and initialize storage space.
        for isag in InStreamAgents:
            self.Q_LSM[isag] = np.zeros(self.WS["DataLength"])
        
        for t in tqdm(range(self.WS["DataLength"]), desc = self.__name__):
            CurrentDate = pdDatedateIndex[t]
            for node in SimSeq:
                
                if node in InStreamAgents:     # The node is an in-stream agent.
                    #----- Update in-stream agent's actions to streamflow (self.Q_LSM) for later routing usage.
                    for ag in AgSimSeq["AgSimPlus"][node]:
                        #self.Q_LSM = Agents[ag].act(self.Q_LSM, StartDate, CurrentDate)      # Define in Basic Agent Class.
                        pass
                    for ag in AgSimSeq["AgSimMinus"][node]:
                        #self.Q_LSM = Agents[ag].act(self.Q_LSM, StartDate, CurrentDate)      # Define in Basic Agent Class.
                        pass
                else:   # The node is a routing outlet.
                    if self.RR["Model"] == "Lohmann":
                        
                        #----- Update none in-stream agent's actions (imply in AgSimSeq calculation) to streamflow for later routing usage.
                        for ag in AgSimSeq["AgSimPlus"][node]:
                            #self.Q_LSM = Agents[ag].act(self.Q_LSM, StartDate, CurrentDate)      # Define in Basic Agent Class.
                            pass
                        for ag in AgSimSeq["AgSimMinus"][node]:
                            #self.Q_LSM = Agents[ag].act(self.Q_LSM, StartDate, CurrentDate)      # Define in Basic Agent Class.
                            pass
                        
                        #----- Run Lohmann routing model for one routing outlet (node) for 1 timestep (day). 
                        Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_LSM, t)
                        
                        #----- Store Qt to final output.
                        self.Q[node][t] = Qt 
        #print("\n")
        logger.info("[{}] Complete HydroCNHS simulation! [{}]\n".format(self.__name__, getElapsedTime()))
        return self.Q





        # for t in range(self.WS["DataLength"]):
        #     CurrentDate = pdDatedateIndex[t]
        #     # ----- Update Qt by ABM
        #     ActiveAgTypes = self.checkActiveAgTypes(StartDate, CurrentDate)
        #     for agType in ActiveAgTypes:
        #         # Assume we have a function called runAgent(agInputs, agPars, Q)
        #         # agParel = Parallel(n_jobs = Paral["ABM"], verbose = Paral["verbose"]) \
        #         #             ( delayed(runAgent)\
        #         #             (self.ABM[agType][ag]["Inputs"], self.ABM[agType][ag]["Inputs"]["Pars"], self.Q_LSM) \
        #         #             for ag in self.ABM[agType] )     
        #         pass
            
        #     # ----- Calculate gauged Qt by routing  
        #     Qt = None
        #     if self.RR["Model"] == "Lohmann":
        #         Qt = runTimeStep_Lohmann(RoutingOutlets, self.RR, self.UH_Lohmann, self.Q_LSM, t)

        #     for g in self.Q:
        #         self.Q[g][t] = Qt[g]