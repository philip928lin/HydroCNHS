# Main HydroCNHS simulator.
# This module is design to couple the human model with semi-distributed 
# hydrological model to form a coupled natural human system.
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# 2021/02/05

from joblib import Parallel, delayed        # For parallelization
from pandas import date_range, to_datetime
from tqdm import tqdm
from copy import Error, deepcopy            # For deepcopy dictionary.
import traceback
import numpy as np
import time
import logging

from .LSM import runGWLF, cal_pet_Hamon, runHYMOD, runABCD
from .Routing import formUH_Lohmann, runTimeStep_Lohmann
from .SystemControl import loadConfig, loadModel, loadCustomizedModule2Class

# The customized agent file will be imported externally
from .Agent_customize3 import *        # AgType_Reservoir, AgType_IrrDiversion

class HydroCNHSModel(object):
    """Main HydroCNHS simulation object.

    """
    def __init__(self, model, name = None):
        """HydroCNHS constructor

        Args:
            model (str/dict): model.yaml file (prefer) or dictionary. 
            name ([str], optional): Object name. Defaults to None.
        """
        # Assign model name and get logger.
        self.__name__ = name
        if name is None:
            self.logger = logging.getLogger("HydroCNHS") # Get logger 
        else:
            self.logger = logging.getLogger("HydroCNHS."+name) # Get logger 
        
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
            self.ABM = Model.get("ABM")        # ABM can be none (None coupled model)
            self.SysPD = Model["SystemParsedData"]
        except:
            self.logger.error("Model file is incomplete for HydroCNHS.")
            
        # Initialize output
        self.Q_routed = {}     # [cms] Streamflow for routing outlets.

    
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
                PE[sb] = cal_pet_Hamon(T[sb], self.LSM[sb]["Inputs"]["Latitude"], self.WS["StartDate"], dz = None)
            self.logger.info("Compute PEt by Hamon method. Users can improve the efficiency by assigning pre-calculated PEt.")
        self.Weather = {"T":T, "P":P, "PE":PE}
        self.logger.info("Load T & P & PE with total length {}.".format(self.WS["DataLength"]))
           
    def __call__(self, T, P, PE = None, AssignedQ = {}, AssignedUH = {}, disable = False):
        """Run HydroCNHS simulation. The simulation is controled by model.yaml and Config.yaml (HydroCNHS system file).
        
        Args:
            T (dict): Daily mean temperature.
            P (dict): Daily precipitation.
            PE (dict, optional): Potential evapotranspiration. Defaults to None (calculted by Hamon's method).
            AssignedQ (dict, optional): If user want to manually assign Q (value, Array) for certain outlet (key, str). Defaults to None.
            AssignedUH (dict, optional): If user want to manually assign UH (Lohmann) (value, Array) for certain outlet (key, str). Defaults to None.
        """
        
        # Start a timer -------------------------------------------------------
        start_time = time.monotonic()
        self.elapsed_time = 0
        def getElapsedTime():
            elapsed_time = time.monotonic() - start_time
            self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            return self.elapsed_time
        
        Paral = self.Config["Parallelization"]
        Outlets = self.WS["Outlets"]
        self.Q_LSM = {}     # Store Q result for land surface simulation.
        
        # ----- Land surface simulation ---------------------------------------
        # Remove sub-basin that don't need to be simulated. 
        # Not preserving element order in the list.
        Outlets = list(set(Outlets) - set(AssignedQ.keys()))  
        if AssignedQ != {}:
            RoutingOutlets = self.SysPD["RoutingOutlets"]
            for ro in RoutingOutlets:
                for sb in self.RR[ro]:
                    if sb in AssignedQ:
                        # No in-grid routing.
                        self.RR[ro][sb]["Pars"]["GShape"] = None  
                        self.RR[ro][sb]["Pars"]["GRate"] = None   
                        self.logger.info("Turn {}'s GShape and GRate to None in the routing setting. There is no in-grid time lag with given observed Q.".format((sb, ro)))
        
        # Start GWLF simulation in parallel.
        if self.LSM["Model"] == "GWLF":
            self.Q_GWLF = {}
            self.logger.info("Start GWLF for {} sub-basins. [{}]".format(len(Outlets), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets)    
            QParel = Parallel(n_jobs = Paral["Cores_LSM"], verbose = Paral["verbose"]) \
                            ( delayed(runGWLF)\
                                (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.Weather["PE"][sb], self.WS["StartDate"], self.WS["DataLength"]) \
                                for sb in Outlets ) 
                            
        # Start HYMOD simulation in parallel.
        # Not verify this model yet.
        if self.LSM["Model"] == "HYMOD":
            self.logger.info("Start HYMOD for {} sub-basins. [{}]".format(len(Outlets), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets)    
            QParel = Parallel(n_jobs = Paral["Cores_LSM"], verbose = Paral["verbose"]) \
                            ( delayed(runHYMOD)\
                                (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.Weather["PE"][sb], self.WS["DataLength"]) \
                                for sb in Outlets ) 
        
        # Start ABCD simulation in parallel.
        # Not verify this model yet.
        if self.LSM["Model"] == "ABCD":
            self.logger.info("Start ABCD for {} sub-basins. [{}]".format(len(Outlets), getElapsedTime()))
            # Load weather and calculate PEt with Hamon's method.
            self.loadWeatherData(T, P, PE, Outlets)    
            QParel = Parallel(n_jobs = Paral["Cores_LSM"], verbose = Paral["verbose"]) \
                            ( delayed(runABCD)\
                                (self.LSM[sb]["Pars"], self.LSM[sb]["Inputs"], self.Weather["T"][sb], self.Weather["P"][sb], self.Weather["PE"][sb], self.WS["DataLength"]) \
                                for sb in Outlets ) 

        # Add user assigned Q first. ------------------------------------------
        self.Q_LSM = deepcopy(AssignedQ)            # Necessary deepcopy!
        # Collect QParel results
        for i, sb in enumerate(Outlets):
            self.Q_LSM[sb] = QParel[i]
            
        # Q_routed will be continuously updated for routing.
        self.Q_routed = deepcopy(self.Q_LSM)        # Necessary deepcopy to isolate self.Q_LSM and self.Q_routed storage pointer!
        self.logger.info("Complete LSM simulation. [{}]".format(getElapsedTime()))
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
            self.logger.info("Start forming {} UHs for Lohmann routing. [{}]".format(len(UH_List_Lohmann), getElapsedTime()))
            UHParel = Parallel(n_jobs = Paral["Cores_formUH_Lohmann"], verbose = Paral["verbose"]) \
                            ( delayed(formUH_Lohmann)\
                            (self.RR[pair[1]][pair[0]]["Inputs"], self.RR[pair[1]][pair[0]]["Pars"]) \
                            for pair in UH_List_Lohmann )     # pair = (Outlet, GaugedOutlet)

            # Form UH ---------------------------------------------------------
            # Add user assigned UH first.
            self.UH_Lohmann = deepcopy(AssignedUH)  # Necessary deepcopy!
            for i, pair in enumerate(UH_List_Lohmann):
                self.UH_Lohmann[pair] = UHParel[i]
            self.logger.info("Complete forming UHs for Lohmann routing. [{}]".format(getElapsedTime()))
        # ---------------------------------------------------------------------

        
        # ----- Load Agents from ABM ------------------------------------------
        # We will automatically detect whether the ABM section is available. 
        # If ABM section is not found, then we consider it as none coupled 
        # model.
        # Technical note:
        #   We will load user-defined modules (e.g., AgentType.py) into 
        # HydroCNHS and store them under the UserModules class. Then, User 
        # object is created for Hydro CNHS to apply those user-defined classes.
        # We use eval() to turn string into python variable.
        #   Detailed instruction for designing proper modules for HydroCNHS, 
        # please check the documentation. Certain protocals have to be followed.
        
        StartDate = to_datetime(self.WS["StartDate"], format="%Y/%m/%d")  
        DataLength = self.WS["DataLength"]
        self.Agents = {}     # Store all agent objects with key = agentname.
        self.DMClasses= {}   # Store all DM function Ex {"DMFunc": DMFunc()}

        if self.ABM is not None: 
            ABM = self.ABM
            # Import user-defined module --------------------------------------
            MPath = self.Path.get("Modules")
            if MPath is not None:
                # User class will store all user-defined modules' classes and 
                # functions.
                class UserModules:
                    pass
                for moduleName in ABM["Inputs"]["Modules"]:
                    loadCustomizedModule2Class(UserModules, moduleName, MPath)
                User = UserModules()  # Create an user object.
            
            # Initialize DMFuncs ----------------------------------------------
            for dmclass in ABM["Inputs"]["DMClasses"]:
                try:        # Try to load from user-defined module first.
                    self.DMClasses[dmclass] = eval("User."+dmclass)(StartDate, DataLength, ABM)
                    self.logger.info("Load {} from the user-defined classes.".format(dmclass))
                except Exception as e:
                    try:    # Detect if it is a built-in class.
                        self.DMClasses[dmclass] = eval(dmclass)(StartDate, DataLength, ABM)
                        self.logger.info("Load {} from the built-in classes.".format(dmclass))
                    except Exception as e:
                        self.logger.error(traceback.format_exc())
                        raise Error("Fail to load {}.\nMake sure the class is well-defined in given modules.".format(dmclass))
                
            # Initialize agent action groups ----------------------------------
            # The agent action groups is different from the DMFunc. Action 
            # group do actions (e.g., divert water) together based on their 
            # original decisions (e.g., diversion request) from DMFunc. 
            # This could be used in a situation, where agents share the water 
            # deficiency together. 
            # AgGroup = {"AgType":{"Name": []}}   (in Model.yaml)
            AgGroup = ABM["Inputs"].get("AgGroup")
            if AgGroup is not None:
                for agType in AgGroup:
                    for agG in AgGroup[agType]:
                        agList = AgGroup[agType][agG]
                        agConfig = {}
                        for ag in agList:
                            agConfig[ag] = ABM[agType][ag]
                        try:      # Try to load from user-defined module first.
                            self.Agents[agG] = eval("User."+agType)(Name=agG, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
                            self.logger.info("Load {} for {} from the user-defined classes.".format(agType, agG))
                        except Exception as e:
                            try:  # Detect if it is a built-in class.
                                self.Agents[agG] = eval(agType)(Name=agG, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
                                self.logger.info("Load {} for {} from the built-in classes.".format(agType, agG))
                            except Exception as e:
                                self.logger.error(traceback.format_exc())
                                raise Error("Fail to load {} for {}.\nMake sure the class is well-defined in given modules.".format(agType, agG))
            else:
                AgGroup = []
                
            # Initialize agents not belong to any action groups ---------------
            for agType, Ags in ABM.items():
                if agType == "Inputs" or agType in AgGroup:
                        continue
                for ag, agConfig in Ags.items():
                    try:        # Try to load from user-defined module first.
                        self.Agents[ag] = eval("User."+agType)(Name=ag, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
                        self.logger.info("Load {} for {} from the user-defined classes.".format(agType, ag))
                    except Exception as e:
                        try:    # Detect if it is a built-in class.
                            self.Agents[ag] = eval(agType)(Name=ag, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
                            self.logger.info("Load {} for {} from the built-in classes.".format(agType, ag))
                        except Exception as e:
                            self.logger.error(traceback.format_exc())
                            raise Error("Fail to load {} for {}.\nMake sure the class is well-defined in given modules.".format(agType, ag))
        # ---------------------------------------------------------------------
        
        
        # ----- Time step simulation (Coupling hydrological model and ABM) ----
        # Obtain datetime index -----------------------------------------------
        pdDatedateIndex = date_range(start = StartDate, periods = DataLength, freq = "D")
        self.pdDatedateIndex = pdDatedateIndex  # So users can use it directly.    
        
        # Load system-parsed data ---------------------------------------------
        SimSeq = self.SysPD["SimSeq"]
        AgSimSeq = self.SysPD["AgSimSeq"]
        InStreamAgents = self.SysPD["InStreamAgents"]   
        
        # Add instream agent to Q_routed --------------------------------------
        # InStreamAgents include ResDamAgentTypes & DamDivAgentTypes
        for isag in InStreamAgents:
            self.Q_routed[isag] = np.zeros(DataLength)
        
        ##### Only a semi-distributed hydrological model ######################
        #####                     (Only LSM and Routing)
        if self.ABM is None: 
            self.logger.info("Start the non-coupled simulation.")
            # Run step-wise routing to update Q_routed ------------------------
            for t in tqdm(range(DataLength), desc = self.__name__, disable=disable):
                CurrentDate = pdDatedateIndex[t]
                for node in SimSeq:
                    if node in RoutingOutlets:
                        #----- Run Lohmann routing model for one routing outlet (node) for 1 timestep (day).
                        if self.RR["Model"] == "Lohmann":
                            Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_routed, self.Q_LSM, t)
                        #----- Store Qt to final output.
                        self.Q_routed[node][t] = Qt 
                        
        ##### HydroCNHS model (Coupled model) #################################
        # We create four interfaces for "two-way coupling" between natural  
        # model and human model. However, the user-defined human model has to 
        # follow specific protocal. See the documantation for details.
        else:
            self.logger.info("Start the HydroCNHS simulation.")
            for t in tqdm(range(DataLength), desc = self.__name__, disable=disable):
                CurrentDate = pdDatedateIndex[t]
                for node in SimSeq:
                    # Load active agent for current node at time t ------------
                    RiverDivAgents_Plus = AgSimSeq["AgSimPlus"][node].get("RiverDivAgents")
                    RiverDivAgents_Minus = AgSimSeq["AgSimMinus"][node].get("RiverDivAgents")
                    InsituDivAgents_Minus = AgSimSeq["AgSimMinus"][node].get("InsituDivAgents")
                    DamDivAgents_Plus = AgSimSeq["AgSimPlus"][node].get("DamDivAgents")
                    ResDamAgents_Plus = AgSimSeq["AgSimPlus"][node].get("ResDamAgents")

                    # Note for the first three if, we should only enter one of 
                    # them at each node.
                    if InsituDivAgents_Minus is not None or RiverDivAgents_Plus is not None:
                        r"""
                        For InsituDivAgents, they divert water directly from 
                        the runoff in each sub-basin or grid.
                        Note that InsituDivAgents has no return flow option.
                        After updating Q generated by LSM and plus the return 
                        flow, we run the routing to calculate the routing 
                        streamflow at the routing outlet stored in Q_routed.
                        Note that we need to use the updated self.Q_LSM = 
                                self.Q_LSM - Div + return flow
                        for the routing outlet to run the routing model.
                        That means return flow will join the in-grid routing!
                        Therefore, in this section, both self.Q_routed and 
                        self.Q_LSM will be updated.
                        """
                        if InsituDivAgents_Minus is not None:
                            for ag in InsituDivAgents_Minus:
                                # self.Q_LSM - Div
                                self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t, DMs = self.DMClasses)
                                self.Q_LSM[node][t] = self.Q_routed[node][t]
                                
                        if RiverDivAgents_Plus is not None:    
                            # e.g., add return flow
                            for ag in RiverDivAgents_Plus:
                                # self.Q_LSM + return flow   
                                # => return flow will join the in-grid routing. 
                                self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t, DMs = self.DMClasses)
                                self.Q_LSM[node][t] = self.Q_routed[node][t]
                    
                    elif ResDamAgents_Plus is not None:
                        r"""
                        For ResDamAgents, we simply add the release water to 
                        self.Q_routed[isag]. No minus action is needed for its 
                        upstream inflow outlet.
                        """
                        for ag in ResDamAgents_Plus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t, DMs = self.DMClasses)
                    
                    elif DamDivAgents_Plus is not None:
                        r"""
                        For DamDivAgents_Plus, we simply add the release water 
                        to self.Q_routed[isag]. No minus action is needed.
                        Note that even the DM is diversion, the action should 
                        be converted to release Q (code in agent class). 
                        Diversion dam.
                        """
                        for ag in DamDivAgents_Plus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t, DMs = self.DMClasses)
                    
                    if node in RoutingOutlets:
                        #----- Run Lohmann routing model for one routing outlet
                        # (node) for 1 time step (day).
                        if self.RR["Model"] == "Lohmann":
                            Qt = runTimeStep_Lohmann(node, self.RR, self.UH_Lohmann, self.Q_routed, self.Q_LSM, t)
                        #----- Store Qt to final output.
                        self.Q_routed[node][t] = Qt 
                        
                    
                    if RiverDivAgents_Minus is not None:
                        r"""
                        For RiverDivAgents_Minus, we divert water from the 
                        routed river flow at agent-associated routing outlet.
                        """
                        for ag in RiverDivAgents_Minus:
                            self.Q_routed = self.Agents[ag].act(self.Q_routed, AgentDict = self.Agents, node=node, CurrentDate=CurrentDate, t=t, DMs = self.DMClasses)
                    

        # ----------------------------------------------------------------------
        print("")   # Force the logger to start a new line after tqdm.
        self.logger.info("Complete HydroCNHS simulation! [{}]\n".format(getElapsedTime()))
        # [cms] Streamflow for routing outlets (Gauged outlets and inflow outlets of in-stream agents).
        # For other variables users need to extract them manually from this class.
        return self.Q_routed   
    
    def getModelObject(self):
        return self.__dict__
    
    
    
r"""
# ----- Load Agents from ABM -----
        # Note: we will automatically detect whether the ABM section is 
        # available. If ABM section is not found, then we consider it as 
        # none coupled model.
        # AgGroup = {"AgType":{"Name": []}}
        
        StartDate = to_datetime(self.WS["StartDate"], format="%Y/%m/%d")  
        DataLength = self.WS["DataLength"]
        self.Agents = {}     # Store all agent objects with key = agentname.

        if self.ABM is not None: 
            #====== Customize part ======
            self.DivDM_KTRWS = DivDM(StartDate, DataLength, self.ABM)
            #============================
            # Create agent group
            AgGroup = self.ABM["Inputs"].get("AgGroup")
            if AgGroup is not None:
                for agType in AgGroup:
                    for agG in AgGroup[agType]:
                        agList = AgGroup[agType][agG]
                        agConfig = {}
                        for ag in agList:
                            agConfig[ag] = self.ABM[agType][ag]
                        self.Agents[agG] = eval(agType)(Name=agG, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
            else:
                AgGroup = []
            # Create agent
            for agType, Ags in self.ABM.items():
                if agType == "Inputs" or agType in AgGroup:
                        continue
                for ag, agConfig in Ags.items():
                    # eval(agType) will turn the string into class. Therefore, agType must be a well-defined class in Agent module.
                    try:
                        # Initialize agent object from agent-type class defined in Agent.py.
                        self.Agents[ag] = eval(agType)(Name=ag, Config=agConfig, StartDate=StartDate, DataLength=DataLength)
                    except Exception as e:
                        self.logger.error(traceback.format_exc())
                        raise Error("Fail to load {} as agent type {}.".format(ag, agType))
        # --------------------------------
"""