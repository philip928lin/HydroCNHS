#%%
# Form the water system using a semi distribution hydrological model and agent-based model.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

from .LSM import runGWLF, calPEt_Hamon
from .RiverRouting import formUH_Lohmann, runTimeStep_Lohmann
from .SystemConrol import loadConfig, loadModel, checkModel
from joblib import Parallel, delayed    # For parallelization
from pandas import date_range, to_datetime
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
            
        # Load model and distribute into several variables.
        if isinstance(model, str):
            Model = loadModel(model)
        else: # If given dictionary directly.
            Model = model    
            checkModel(Model)  
                           
        self.Path = Model["Path"]
        self.WS = Model["WaterSystem"]     # WS: Water system
        self.LSM = Model["LSM"]     # HP: Hydrological process
        self.RR = Model["RiverRouting"]     # RR: River routing 
        self.ABM = Model["ABM"] 

        # initialize output
        self.Q = {}     # [cms] Streamflow for each outlet.
        for g in self.WS["GaugedOutlets"]:
            self.Q[g] = np.zeros(self.WS["DataLength"])
        self.A = {}     # Collect output for AgentType Agent its outputs
    
    def loadWeatherData(self, T, P, PE, OutletsQ = None):
        """Load temperature and precipitation data.
        CAn add some check function or deal with miss value here.
        Args:
            T (dict): [degC] Daily mean temperature time series data (value) for each sub-basin named by its outlet.
            P (dict): [cm] Daily precipitation time series data (value) for each sub-basin named by its outlet.
            PE(dict/None): [cm] Daily potential evapotranpiration time series data (value) for each sub-basin named by its outlet.
        """
        if PE is None:
            PE = {}
            # Default to calculate PE with Hamon's method and no dz adjustment.
            for sb in OutletsQ:
                PE[sb] = calPEt_Hamon(T[sb], self.LSM[sb]["Input"]["Latitude"], self.WS["StartDate"], dz = None)
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
        
    def run(self, T, P, PE = None, AssignedQ = None, AssignedUH = None):
        """Run HydroCNHS simulation. The simulation is controled by model.yaml and Config.yaml (HydroCNHS system file).
        
        Args:
            T (dict): Daily mean temperature.
            P (dict): Daily precipitation.
            PE (dict, optional): Potential evapotranspiration. Defaults to None (calculted by Hamon's method).
            AssignedQ (dict, optional): If user want to manually assign Q (value, Array) for certain outlet (key, str). Defaults to None.
            AssignedUH (dict, optional): [description]. If user want to manually assign UH (Lohmann) (value, Array) for certain outlet (key, str). Defaults to None.
        """
        
        # Set a timer here
        start_time = time.monotonic()
        self.elapsed_time = 0
        
        Para = self.Config["Parallelization"]
        Outlets = self.WS["Outlets"]
        self.Q_HP = {}  # Temporily store Q result from land surface simulation.
        
        # ----- Land surface simulation ---------------------------------------
        if self.LSM["Model"] == "GWLF":
            # Remove sub-basin that don't need to be simulated. Not preserving element order in the list.
            Outlets_GWLF = list(set(Outlets) - set(AssignedQ.keys()))  
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets_GWLF)    
            # Start GWLF simulation in parallel.
            QParel = Parallel(n_jobs = Para["Cores_runGWLF"], verbose = Para["verbose"]) \
                            ( delayed(runGWLF)\
                              (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.PE[sb], self.WS["StartDate"], self.WS["DataLength"]) \
                              for sb in Outlets_GWLF ) 
            # Add user assigned Q first.
            self.Q_HP = AssignedQ      
            # Collect QParel results
            for i, sb in enumerate(Outlets_GWLF):
                self.Q_HP[sb] = QParel[i]
        # ---------------------------------------------------------------------    
    

        # ----- Form UH for Lohmann routing method ----------------------------
        if self.RR["Model"] == "Lohmann":
            self.UH_Lohmann = {}    # Initialized the output dictionary
            # Form combination
            UH_List = [(sb, g) for g in self.WS["GaugedOutlets"] for sb in self.RR["g"]]
            # Remove assigned UH from the list. Not preserving element order in the list.
            UH_List_Lohmann = list(set(UH_List) - set(AssignedUH.keys()))
            # Start forming UH_Lohmann in parallel.
            UHParel = Parallel(n_jobs = Para["Cores_formUH_Lohmann"], verbose = Para["verbose"]) \
                            ( delayed(formUH_Lohmann)\
                            (self.RR[pair[1]][pair[0]]["Inputs"]["Flowlength"], self.RR[pair[1]][pair[0]]["Pars"]) \
                            for pair in UH_List_Lohmann )     # pair = (Outlet, GaugedOutlet)
            # Add user assigned UH first.

            # Form UH
            self.UH_Lohmann = AssignedUH
            for i, pair in enumerate(UH_List_Lohmann):
                self.UH_Lohmann[pair] = UHParel[i]
        # ---------------------------------------------------------------------

        # ----- Time step simulation (Coupling hydrological model and ABM)-----
        # Note: Only Qt of GaugedOutlet1 and the 
        # Obtain datetime index
        StartDate = to_datetime(self.WS["StartDate"], format="%Y/%m/%d")        
        pdDatedateIndex = date_range(start = StartDate, periods = self.WS["DataLength"], freq = "D")    
                        
        for t in (self.WS["DataLength"]):
            CurrentDate = pdDatedateIndex[t]
            # ----- Update Qt by ABM
            ActiveAgTypes = self.checkActiveAgTypes(self, t, StartDate, CurrentDate)
            for agType in ActiveAgTypes:
                # Assume we have a function called runAgent(agInputs, agPars, Q)
                # agParel = Parallel(n_jobs = Para["ABM"], verbose = Para["verbose"]) \
                #             ( delayed(runAgent)\
                #             (self.ABM[agType][ag]["Inputs"], self.ABM[agType][ag]["Inputs"]["Pars"], self.Q_HP) \
                #             for ag in self.ABM[agType] )     
                pass
            
            # ----- Calculate gauged Qt by routing  
            Qt = None
            if self.RR["Model"] == "Lohmann":
                Qt = runTimeStep_Lohmann(self.WS["GaugedOutlets"], self.RR, self.UH_Lohmann, self.Q_HP, t)
            
            # ----- Store Qt
            for g in self.Q:
                self.Q[g][t] = Qt[g]
            
            
        elapsed_time = time.monotonic() - start_time
        self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        logger.info("Complete HydroCNHS simulation [{}].".format(self.elapsed_time))
        return 